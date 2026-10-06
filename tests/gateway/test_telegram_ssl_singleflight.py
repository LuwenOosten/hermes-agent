"""Single-flight coverage for the shared Telegram TLS context (PR #133340 review).

``shared_ssl_context()`` used to run ``asyncio.to_thread(_shared_ssl_context_sync)`` on
every cold call, and the sync builder serialized on a threading lock. Each caller a
connect deadline cancelled left its default-executor worker parked on that lock; two of
them exhausted a two-worker executor and starved unrelated ``asyncio.to_thread`` work,
even though no caller was still waiting for the context.

The rework runs one process-wide daemon worker behind one native
``concurrent.futures.Future``. Cancelled callers share that Future, cannot cancel it, and
never occupy an executor slot. These tests drive the real event loop and threading
primitives with Events only; every worker started is released and joined before its
monkeypatch is restored.
"""

import asyncio
import concurrent.futures
import contextvars
import gc
import ssl
import threading
from types import SimpleNamespace

import pytest

import agent.ssl_verify as ssl_verify
from agent import memory_provider
import plugins.platforms.telegram.telegram_network as tnet

_REAL_CTX = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


class _BlockingFactory:
    """Stand-in for ``agent.ssl_verify.platform_ssl_context`` that stalls on an Event.

    ``started`` fires when the factory is entered, ``release`` gates its return/raise.
    ``calls``/``threads`` record every build so a test can prove one build happened on a
    worker thread and the warm path rebuilt nothing.
    """

    def __init__(self, *, result=None, error=None):
        self.result = result
        self.error = error
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.threads: list = []

    def __call__(self):
        self.calls += 1
        self.threads.append(threading.current_thread())
        self.started.set()
        assert self.release.wait(timeout=5.0), "test never released the shared-context factory"
        if self.error is not None:
            raise self.error
        return self.result


class _UnstartableThread:
    """What the real spawn helper hands back when the OS refuses another thread."""

    def start(self):
        raise RuntimeError("can't start new thread")


@pytest.fixture(autouse=True)
def tls_env(monkeypatch):
    """Isolate the process-wide TLS state and join every worker before teardown."""
    workers: list = []
    real_spawn = memory_provider.spawn_context_thread

    def _record_worker(*args, **kwargs):
        thread = real_spawn(*args, **kwargs)
        workers.append(thread)
        return thread

    monkeypatch.setattr(memory_provider, "spawn_context_thread", _record_worker)
    monkeypatch.setattr(tnet, "_SSL_CONTEXT", None)
    monkeypatch.setattr(tnet, "_SSL_CONTEXT_FUTURE", None)
    yield SimpleNamespace(workers=workers, spawn=_record_worker, real_spawn=real_spawn)
    for thread in workers:
        thread.join(timeout=5.0)
    assert not any(thread.is_alive() for thread in workers), "a shared-context worker outlived its test"


async def _cancel_and_drain(*waiters):
    """Cancel any still-pending waiter and consume its outcome (from a test finally)."""
    pending = [waiter for waiter in waiters if waiter is not None and not waiter.done()]
    for waiter in pending:
        waiter.cancel()
    await asyncio.gather(*(waiter for waiter in waiters if waiter is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_cold_waiters_leave_default_executor_free(monkeypatch, tls_env):
    """Repeated cancelled cold callers must not park the two default-executor workers.

    With the old ``asyncio.to_thread`` path the first cancelled caller left one worker
    inside the factory and the second left one on the factory lock, so the unrelated
    ``to_thread`` below could never run while the factory stayed blocked.
    """
    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=2))
    factory = _BlockingFactory(result=_REAL_CTX)
    monkeypatch.setattr(ssl_verify, "platform_ssl_context", factory)

    first = None
    cancelled = []
    try:
        first = asyncio.ensure_future(tnet.shared_ssl_context())
        assert await asyncio.wait_for(asyncio.to_thread(factory.started.wait), timeout=2.0)
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)

        for _ in range(4):
            task = asyncio.ensure_future(tnet.shared_ssl_context())
            await asyncio.sleep(0)  # let the task reach its shared-future await
            task.cancel()
            cancelled.append(task)
        await asyncio.gather(*cancelled, return_exceptions=True)

        assert await asyncio.wait_for(asyncio.to_thread(lambda: "unrelated work"), timeout=2.0) == "unrelated work"
        assert factory.calls == 1, "cold callers rebuilt the context instead of sharing one build"
        assert factory.threads[0] is not threading.main_thread(), "the build ran on the event-loop thread"
    finally:
        await _cancel_and_drain(first, *cancelled)
        factory.release.set()


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_the_shared_build(monkeypatch, tls_env):
    """One waiter cancelling leaves the other waiter's pending build untouched."""
    factory = _BlockingFactory(result=_REAL_CTX)
    monkeypatch.setattr(ssl_verify, "platform_ssl_context", factory)

    first = second = None
    try:
        first = asyncio.ensure_future(tnet.shared_ssl_context())
        assert await asyncio.wait_for(asyncio.to_thread(factory.started.wait), timeout=2.0)
        second = asyncio.ensure_future(tnet.shared_ssl_context())
        await asyncio.sleep(0)  # second joins the same in-flight initialization

        first.cancel()
        assert isinstance((await asyncio.gather(first, return_exceptions=True))[0], asyncio.CancelledError)

        factory.release.set()
        assert await asyncio.wait_for(second, timeout=2.0) is _REAL_CTX

        # Warm path: same identity, no factory call, no second worker.
        assert await asyncio.wait_for(tnet.shared_ssl_context(), timeout=2.0) is _REAL_CTX
        assert factory.calls == 1
        assert len(tls_env.workers) == 1, "a cancelled waiter started a second initialization worker"
    finally:
        await _cancel_and_drain(first, second)
        factory.release.set()


def test_second_loop_joins_the_in_flight_build_after_the_first_loop_closes(monkeypatch, tls_env):
    """A loop that closes after cancelling still leaves a joinable in-flight build."""
    factory = _BlockingFactory(result=_REAL_CTX)
    monkeypatch.setattr(ssl_verify, "platform_ssl_context", factory)

    first_loop = asyncio.new_event_loop()
    second_loop = asyncio.new_event_loop()
    runner = None
    second_thread: list = []
    joined = threading.Event()
    result: dict = {}
    real_admit = tnet._shared_ssl_context_future

    def _recording_admit():
        future = real_admit()
        if second_thread and threading.current_thread() is second_thread[0]:
            joined.set()
        return future

    monkeypatch.setattr(tnet, "_shared_ssl_context_future", _recording_admit)

    def _drain_loop(loop):
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.run_until_complete(loop.shutdown_default_executor())

    try:
        first_loop.create_task(tnet.shared_ssl_context())
        assert first_loop.run_until_complete(
            asyncio.wait_for(asyncio.to_thread(factory.started.wait), timeout=2.0))
        _drain_loop(first_loop)
        first_loop.close()

        def _run_second_loop():
            second_thread.append(threading.current_thread())
            try:
                result["ctx"] = second_loop.run_until_complete(
                    asyncio.wait_for(tnet.shared_ssl_context(), timeout=2.0))
            finally:
                second_loop.close()

        runner = threading.Thread(target=_run_second_loop, name="tls-test-second-loop")
        runner.start()
        assert joined.wait(timeout=2.0), "the second loop never joined the in-flight initialization"
        assert len(tls_env.workers) == 1, "the second loop started another initialization worker"

        factory.release.set()
    finally:
        factory.release.set()
        if runner is not None:
            runner.join(timeout=5.0)
        if not second_loop.is_closed() and (runner is None or not runner.is_alive()):
            second_loop.close()
        if not first_loop.is_closed():
            _drain_loop(first_loop)
            first_loop.close()

    assert runner is not None and not runner.is_alive()
    assert result["ctx"] is _REAL_CTX
    assert factory.calls == 1


@pytest.mark.asyncio
async def test_failure_fans_out_and_a_later_call_retries(monkeypatch, tls_env):
    """The failure reaches every waiter and is not sticky for the next cold call."""
    failure = OSError("trust store unavailable")
    failing = _BlockingFactory(error=failure)
    monkeypatch.setattr(ssl_verify, "platform_ssl_context", failing)

    first = second = None
    try:
        first = asyncio.ensure_future(tnet.shared_ssl_context())
        assert await asyncio.wait_for(asyncio.to_thread(failing.started.wait), timeout=2.0)
        second = asyncio.ensure_future(tnet.shared_ssl_context())
        await asyncio.sleep(0)
        failing.release.set()

        results = await asyncio.gather(first, second, return_exceptions=True)
        assert [type(result) for result in results] == [OSError, OSError]
        assert all(result is failure for result in results), "waiters did not share the factory failure"
        assert failing.calls == 1, "the failed initialization ran more than once"

        succeeding = _BlockingFactory(result=_REAL_CTX)
        succeeding.release.set()
        monkeypatch.setattr(ssl_verify, "platform_ssl_context", succeeding)
        assert await asyncio.wait_for(tnet.shared_ssl_context(), timeout=2.0) is _REAL_CTX
        assert succeeding.calls == 1, "the retry did not run a fresh initialization"
    finally:
        await _cancel_and_drain(first, second)
        failing.release.set()


@pytest.mark.asyncio
async def test_cancelled_waiters_leave_no_unhandled_future_reports(monkeypatch, tls_env):
    """All waiters cancelling first must not turn the later failure into loop reports.

    ``asyncio.shield`` on Python 3.14 reports an abandoned shielded future's exception
    through the loop exception handler ("exception in shielded future"), so the shared
    future is awaited directly and its wrapper is cancelled with the caller.
    """
    loop = asyncio.get_running_loop()
    reports: list = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reports.append(context))

    factory = _BlockingFactory(error=OSError("trust store unavailable"))
    monkeypatch.setattr(ssl_verify, "platform_ssl_context", factory)

    waiters: list = []
    try:
        waiters = [asyncio.ensure_future(tnet.shared_ssl_context()) for _ in range(3)]
        assert await asyncio.wait_for(asyncio.to_thread(factory.started.wait), timeout=2.0)
        for waiter in waiters:
            waiter.cancel()
        await asyncio.gather(*waiters, return_exceptions=True)

        factory.release.set()
        for worker in tls_env.workers:
            worker.join(timeout=5.0)
        assert not any(worker.is_alive() for worker in tls_env.workers), "the failed initialization worker did not finish"

        # The worker has now failed and exited; only then drain the abandoned wrappers and
        # force collection, so a dropped unretrieved failure is reported while we watch.
        for _ in range(3):
            await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)

        assert reports == [], f"cancelled waiters left asyncio exception reports: {reports}"

        succeeding = _BlockingFactory(result=_REAL_CTX)
        succeeding.release.set()
        monkeypatch.setattr(ssl_verify, "platform_ssl_context", succeeding)
        assert await asyncio.wait_for(tnet.shared_ssl_context(), timeout=2.0) is _REAL_CTX
        assert succeeding.calls == 1
    finally:
        await _cancel_and_drain(*waiters)
        factory.release.set()
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_cancellation_racing_a_failed_wrapper_stays_quiet(monkeypatch, tls_env):
    """A wrapper that fails in the turn its caller is cancelled must stay quiet.

    The shared helper installs a done callback on its own ``asyncio.wrap_future`` wrapper
    that marks a failure retrieved; without it, cancelling the awaiting task after the
    wrapper already holds an exception would leave an unretrieved Future behind. The
    public ``asyncio.wrap_future`` seam is wrapped only to capture the standard asyncio
    Future the real helper created.
    """
    loop = asyncio.get_running_loop()
    reports: list = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: reports.append(context))

    native = concurrent.futures.Future()
    native.set_running_or_notify_cancel()
    captured: dict = {}
    real_wrap_future = asyncio.wrap_future

    def _capturing_wrap_future(source, **kwargs):
        wrapped = real_wrap_future(source, **kwargs)
        captured["wrapped"] = wrapped
        return wrapped

    monkeypatch.setattr(tnet, "_shared_ssl_context_future", lambda: native)
    monkeypatch.setattr(asyncio, "wrap_future", _capturing_wrap_future)

    waiter = None
    try:
        waiter = asyncio.ensure_future(tnet.shared_ssl_context())
        await asyncio.sleep(0)
        wrapped = captured["wrapped"]
        assert not wrapped.done()

        # Same event-loop turn: the build fails, then the waiter is cancelled before the
        # loop runs either future's callbacks.
        native.set_exception(OSError("build failed"))
        assert wrapped.done()
        waiter.cancel()
        assert isinstance((await asyncio.gather(waiter, return_exceptions=True))[0], asyncio.CancelledError)

        native = None
        wrapped = None
        captured.clear()
        for _ in range(3):
            await asyncio.sleep(0)
        gc.collect()
        await asyncio.sleep(0)
        assert reports == [], f"a cancelled wrapper reported an unretrieved failure: {reports}"
    finally:
        await _cancel_and_drain(waiter)
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_thread_start_failure_does_not_pin_a_pending_initialization(monkeypatch, tls_env):
    """A refused thread start reports to the caller and leaves the next call retryable."""
    monkeypatch.setattr(ssl_verify, "platform_ssl_context", lambda: _REAL_CTX)

    def _refuse_thread_start(*args, **kwargs):
        return _UnstartableThread()

    monkeypatch.setattr(memory_provider, "spawn_context_thread", _refuse_thread_start)

    with pytest.raises(RuntimeError, match="can't start new thread"):
        await tnet.shared_ssl_context()
    assert tnet._SSL_CONTEXT_FUTURE is None

    monkeypatch.setattr(memory_provider, "spawn_context_thread", tls_env.spawn)
    assert await asyncio.wait_for(tnet.shared_ssl_context(), timeout=2.0) is _REAL_CTX


_PROBE = contextvars.ContextVar("tls_factory_probe", default="unset")


@pytest.mark.asyncio
async def test_factory_inherits_the_first_callers_contextvars(monkeypatch, tls_env):
    """The worker runs under the caller's contextvars (profile/secret scope)."""
    seen: list = []
    started = threading.Event()
    release = threading.Event()

    def _context_factory():
        seen.append(_PROBE.get())
        started.set()
        assert release.wait(timeout=5.0)
        return _REAL_CTX

    monkeypatch.setattr(ssl_verify, "platform_ssl_context", _context_factory)

    async def _caller():
        _PROBE.set("tenant-alpha")
        return await tnet.shared_ssl_context()

    waiter = None
    try:
        waiter = asyncio.ensure_future(_caller())
        assert await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=2.0)
        release.set()
        assert await asyncio.wait_for(waiter, timeout=2.0) is _REAL_CTX
    finally:
        await _cancel_and_drain(waiter)
        release.set()
    assert seen == ["tenant-alpha"]
