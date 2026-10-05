"""Host-scoped update→restart obligation for ``hermes update``.

Multiplex-only (Teknium ruling): exactly ONE ``hermes gateway run`` per host serves every
profile, so "this pull still owes the fleet a restart" is a property of the HOST, not of one
profile's ``HERMES_HOME``. The legacy ``$HERMES_HOME/fleet_restart_pending`` marker was
per-home: ``hermes -p coder update`` armed and cleared coder's copy while restarting the
SHARED process, and every other profile's CLI could neither see nor discharge that obligation
— it simply armed its own and re-killed the same host process.

The record therefore lives beside the host rendezvous record, in
:func:`gateway.host_rendezvous.host_state_dir` (``$HERMES_GATEWAY_LOCK_DIR`` else
``$XDG_STATE_HOME/hermes/gateway-locks``) — the one cross-profile, per-OS-user state root the
tree already has. It is written once per host, read by every profile's CLI, and cleared once.

The same "one host process, not one per profile" identity is what
:func:`collapse_units_to_host_processes` applies to enumerated systemd units: leftover
per-profile ``hermes-gateway-<p>.service`` units on a multiplexed host all point at the same
live ``MainPID``, so restarting each one restarts the host process N times.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("hermes_cli.update_cmd")

#: One file per OS user, beside ``host-gateway.json`` / ``host-serve.json``.
HOST_OBLIGATION_NAME = "host-update-restart.json"

_RECORD_VERSION = 1


def host_obligation_path() -> Path:
    """Path of the host obligation record."""
    # Not ``gateway.host_rendezvous.host_state_dir``: ``gateway.status`` imports ``utils`` -> ruamel,
    # absent from the historical interpreter an old updater's takeover arms this record in; the
    # recovery module's stdlib copy of the rule is drift-tested against the gateway resolver.
    from hermes_cli.update_restart_recovery import _host_state_dir

    return Path(_host_state_dir()) / HOST_OBLIGATION_NAME


def _write_record(path: Path, record: dict) -> None:
    """``utils.atomic_json_write(path, record, mode=0o600)`` in stdlib only (see above): the temp
    file is 0600 from ``mkstemp``, fsynced, then renamed over the record."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def read_host_obligation() -> Optional[dict]:
    """The published obligation record, or ``None`` when absent/corrupt/foreign-versioned."""
    try:
        payload = json.loads(host_obligation_path().read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != _RECORD_VERSION:
        return None
    return payload


def host_obligation_present() -> bool:
    """True when the record FILE exists, parseable or not.

    Fail-closed: a corrupt record is an obligation whose terms are unknown, never a discharged
    one — the restart is still owed and the reader falls back to "no recorded inventory".
    """
    try:
        return host_obligation_path().is_file()
    except OSError:
        return False


def amend_host_obligation(**fields: Any) -> None:
    """Merge ``fields`` into the armed record (test/diagnostic surface). Never raises."""
    _update_record(lambda record: record.update(fields), "amend")


def write_host_obligation(
    *, expected_sha: str = "", runtimes: Optional[list] = None, profile: str = "", owner: str = ""
) -> bool:
    """Arm the host obligation. True when it was written. Never raises.

    Re-arming from a second profile for the SAME pulled SHA keeps the existing record (and its
    ``restarted`` proof) instead of resetting it: the host owes one restart, not one per profile.
    ``owner`` (an update run's commit-point token) joins the record's ``owners``: a run that
    fails before its move hands back only its own stake (``release_host_obligation``), never
    another install's debt for the same SHA. The first owner also stores what it found there
    (``found``), which the last owner to leave puts back; an owner retargeting to another SHA
    keeps that baseline (``_baseline_for``).
    """
    path = host_obligation_path()
    try:
        with _record_mutex(path):
            return _write_locked(path, expected_sha=expected_sha, runtimes=runtimes, profile=profile, owner=owner)
    except OSError as exc:  # the mutex busy or unopenable: the caller falls back to the per-home marker
        logger.debug("Could not lock the host update-restart obligation: %s", exc)
        return False


def _write_locked(path: Path, *, expected_sha: str, runtimes: Optional[list], profile: str, owner: str) -> bool:
    existing = read_host_obligation()
    try:
        found = _baseline_for(owner, existing, path) if owner else None
    except OSError as exc:  # unreadable is not absent: a guessed ``found`` would delete it later
        logger.debug("Could not read the host update-restart obligation: %s", exc)
        return False
    if existing is not None and expected_sha and existing.get("expected_sha") == expected_sha:
        # Same pull, second profile: the host owes ONE restart, so keep the standing record (and
        # any proof that the restart already happened) rather than resetting it. A later arm that
        # carries the owed inventory still upgrades it — an inventory-less record owes no set.
        fields: dict[str, Any] = {}
        owners = _owners(existing)
        if owner and owner not in owners:
            fields["owners"] = [*owners, owner]
            if not owners:
                fields["found"] = found
        inventory = {"version": 1, "runtimes": runtimes}
        if runtimes is not None and existing.get("inventory") != inventory:
            fields["inventory"] = inventory
        if not fields:
            return True
        try:
            _write_record(path, {**existing, **fields})
        except Exception as exc:  # health: allow BLE001 -- never raises: the caller falls back to the per-home marker
            logger.debug("Could not amend host update-restart obligation: %s", exc)
            return False
        return True
    payload: dict[str, Any] = {
        "version": _RECORD_VERSION,
        "started": time.time(),
        "pid": os.getpid(),
        "armed_by_profile": profile or "",
        "expected_sha": expected_sha or "",
    }
    if owner:
        payload["owners"] = [owner]
        payload["found"] = found
    if runtimes is not None:
        payload["inventory"] = {"version": 1, "runtimes": runtimes}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_record(path, payload)
    except Exception as exc:  # health: allow BLE001 -- never raises: the caller falls back to the per-home marker
        logger.debug("Could not write host update-restart obligation: %s", exc)
        return False
    return True


@contextlib.contextmanager
def _record_mutex(path: Path):
    """Serialize every read-judge-write of the record across processes (``update_lock.marker_mutex``).

    The sidecar sits in a subdirectory: the state dir is the gateway lock dir, whose ``--replace``
    cleanup unlinks every top-level ``*.lock``, and a deleted sidecar splits the lock in two.
    """
    from hermes_cli.update_lock import marker_mutex  # stdlib-only, like this module

    guard = path.parent / f".{path.stem}.mutex"
    guard.mkdir(parents=True, exist_ok=True)
    with marker_mutex(guard / "record"):
        yield


def _owners(record: dict) -> list[str]:
    owners = record.get("owners")
    return [str(o) for o in owners] if isinstance(owners, list) else []


def _baseline_for(owner: str, existing: Optional[dict], path: Path) -> Optional[str]:
    """What ``owner``'s release must put back (base64; ``None`` = absent): the record as it stood
    before ``owner``'s first stake in it (review R1).

    A run re-arms with one owner as it retargets (CP0 branch, origin pull, upstream fork ff), so
    "the record already has owners" never means "nothing was here": a new owner keeps the bytes
    there now, other owners' stakes included; an owner already in the record carries the baseline
    it saved then, or, when it shared the record with others, the shared record minus its own stake.
    """
    owners = _owners(existing or {})
    if owner not in owners:
        return _found_field(path)
    others = [o for o in owners if o != owner]
    if not others:
        return (existing or {}).get("found")
    shared = json.dumps({**(existing or {}), "owners": others}, indent=2, ensure_ascii=False)
    return base64.b64encode(shared.encode("utf-8")).decode("ascii")


def _found_field(path: Path) -> Optional[str]:
    """The record's current bytes (base64), ``None`` when absent: what a first owner puts back.
    Any other read error raises: custody is never guessed."""
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except FileNotFoundError:
        return None


def release_host_obligation(owner: str) -> None:
    """Hand back ``owner``'s stake in the record. When no other run still owes through it, the
    record goes back to what its first owner found (``found``; absent = unlinked).

    A record without ``owner`` was rewritten since (another install's newer pull): it is theirs and
    stays. Raises OSError when the record cannot be rewritten (the caller keeps the debt armed).
    Judged and rewritten under ``_record_mutex``, which every arm takes too, so no arm can land
    between the look and the write and be undone by it (kshitijk4poor F22/N05).
    """
    if not owner or owner not in _owners(read_host_obligation() or {}):
        return  # nothing of ours to hand back: no lock taken, no state dir created
    path = host_obligation_path()
    with _record_mutex(path):
        _release_locked(path, owner)


def _release_locked(path: Path, owner: str) -> None:
    record = read_host_obligation()
    owners = _owners(record) if record is not None else []
    if owner not in owners:
        return  # rewritten since the unlocked look (another install's arm): theirs now
    remaining = [o for o in owners if o != owner]
    if remaining:
        _write_record(path, {**record, "owners": remaining})
        return
    try:
        found = base64.b64decode(record["found"], validate=True) if record.get("found") is not None else None
    except (TypeError, ValueError):
        return  # a damaged ``found`` cannot be put back: the debt stays armed, never guessed away
    if found is None:
        path.unlink(missing_ok=True)
    else:
        replace_bytes(path, found)


def replace_bytes(path: Path, data: bytes) -> None:
    """Put ``data`` back at ``path``: a fresh ``mkstemp`` file (never through a planted alias),
    fsynced, then renamed over the record."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}_", suffix=".restore")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def clear_host_obligation() -> None:
    """Discharge the obligation for the whole host. Never raises."""
    try:
        host_obligation_path().unlink(missing_ok=True)
    except OSError as exc:
        logger.debug("Could not clear host update-restart obligation: %s", exc)


def obligation_fields() -> Optional[dict[str, str]]:
    """The obligation in the legacy ``key=value`` field shape, or ``None`` when unarmed.

    Keeps one parser for both sources: the fields a reader needs (``expected_sha``, the
    serialized ``inventory``) are identical whether they came from the host record or from an
    in-flight legacy per-home marker.
    """
    record = read_host_obligation()
    if record is None:
        return None
    fields = {"expected_sha": str(record.get("expected_sha") or "")}
    inventory = record.get("inventory")
    if inventory is not None:
        fields["inventory"] = json.dumps(inventory)
    return fields


def mark_host_restart_completed(sha: str) -> None:
    """Record that the host process was restarted onto ``sha``. Never raises."""
    _update_record(
        lambda record: record.update(restarted={"sha": sha or "", "pid": os.getpid(), "at": time.time()}),
        "stamp restart completion on")


def _update_record(change: Callable[[dict], None], what: str) -> None:
    """Read, ``change`` and rewrite an armed record under ``_record_mutex`` (an arm's new owner
    stake is never overwritten by a stale copy). Never raises; no record, no write."""
    if read_host_obligation() is None:
        return
    path = host_obligation_path()
    try:
        with _record_mutex(path):
            record = read_host_obligation()
            if record is None:
                return
            change(record)
            _write_record(path, record)
    except Exception as exc:  # health: allow BLE001 -- never raises: a lost stamp costs a repeat restart, never the debt
        logger.debug("Could not %s host update-restart obligation: %s", what, exc)


def host_restart_already_completed(sha: Optional[str]) -> bool:
    """True when THIS host obligation was already restarted onto ``sha``.

    The guard that makes the catch-up restart idempotent per host: a second profile running
    ``hermes update`` must attach to the first restart's outcome, never kill the shared
    multiplexer again.
    """
    record = read_host_obligation()
    if record is None or not sha:
        return False
    restarted = record.get("restarted")
    return isinstance(restarted, dict) and str(restarted.get("sha") or "") == sha


def collapse_units_to_host_processes(
    units: Iterable[str], main_pid: Callable[[str], int]
) -> tuple[list[str], dict[str, str]]:
    """Split enumerated units into ``(restart, {legacy_unit: covering_unit})``.

    Units resolving to the same live ``MainPID`` are ONE host process; restarting each of them
    restarts that process N times, which on a multiplexed host is an N-fold outage triggered by
    leftover per-profile units. A unit with no readable main PID (inactive, unprivileged scope)
    keeps its own restart: identity that cannot be proved is never collapsed away.
    """
    restart: list[str] = []
    covered: dict[str, str] = {}
    owner_by_pid: dict[int, str] = {}
    for unit in units:
        try:
            pid = int(main_pid(unit) or 0)
        except Exception:
            # Identity that cannot be proved keeps its own restart; a probe failure of any kind
            # must never abort the whole pass.
            pid = 0
        if pid <= 0:
            restart.append(unit)
            continue
        owner = owner_by_pid.get(pid)
        if owner is None:
            owner_by_pid[pid] = unit
            restart.append(unit)
        else:
            covered[unit] = owner
    return restart, covered
