"""Tests for plugins.platforms.telegram.telegram_network – fallback transport layer.

Background
----------
api.telegram.org resolves to an IP (e.g. 149.154.166.110) that is unreachable
from some networks.  The workaround: route TCP through a different IP in the
same Telegram-owned 149.154.160.0/20 block (e.g. 149.154.167.220) while
keeping TLS SNI and the Host header as api.telegram.org so Telegram's edge
servers still accept the request.  This is the programmatic equivalent of:

    curl --resolve api.telegram.org:443:149.154.167.220 https://api.telegram.org/bot<token>/getMe

The TelegramFallbackTransport implements this: try known IPv4 Telegram API
IPs first (so a blackholed IPv6 AAAA for the hostname cannot hang
initialize — #87015), then fall through to the dual-stack hostname last,
and "stick" to whichever path works.
"""

import httpx
import pytest
import socket

import plugins.platforms.telegram.telegram_network as tnet

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeTransport(httpx.AsyncBaseTransport):
    """Records calls and raises / returns based on a host→action mapping."""

    def __init__(self, calls, behavior):
        self.calls = calls
        self.behavior = behavior
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(
            {
                "url_host": request.url.host,
                "host_header": request.headers.get("host"),
                "sni_hostname": request.extensions.get("sni_hostname"),
                "path": request.url.path,
            }
        )
        action = self.behavior.get(request.url.host, "ok")
        if action == "timeout":
            raise httpx.ConnectTimeout("timed out")
        if action == "connect_error":
            raise httpx.ConnectError("connect error")
        if isinstance(action, Exception):
            raise action
        return httpx.Response(200, request=request, text="ok")

    async def aclose(self) -> None:
        self.closed = True

def _fake_transport_factory(calls, behavior):
    """Returns a factory that creates FakeTransport instances."""
    instances = []

    def factory(**kwargs):
        t = FakeTransport(calls, behavior)
        instances.append(t)
        return t

    factory.instances = instances
    return factory

def _telegram_request(path="/botTOKEN/getMe"):
    return httpx.Request("GET", f"https://api.telegram.org{path}")

# ═══════════════════════════════════════════════════════════════════════════
# IP parsing & validation
# ═══════════════════════════════════════════════════════════════════════════

class TestParseFallbackIpEnv:
    def test_filters_invalid_and_ipv6(self):
        ips = tnet.parse_fallback_ip_env("149.154.167.220, bad, 2001:67c:4e8:f004::9,149.154.167.220")
        assert ips == ["149.154.167.220", "149.154.167.220"]

    def test_none_returns_empty(self):
        assert tnet.parse_fallback_ip_env(None) == []

# ═══════════════════════════════════════════════════════════════════════════
# Request rewriting
# ═══════════════════════════════════════════════════════════════════════════

class TestRewriteRequestForIp:
    def test_preserves_host_and_sni(self):
        request = _telegram_request()
        rewritten = tnet._rewrite_request_for_ip(request, "149.154.167.220")

        assert rewritten.url.host == "149.154.167.220"
        assert rewritten.headers["host"] == "api.telegram.org"
        assert rewritten.extensions["sni_hostname"] == "api.telegram.org"
        assert rewritten.url.path == "/botTOKEN/getMe"

# ═══════════════════════════════════════════════════════════════════════════
# Fallback transport – core behavior
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackTransport:
    """IPv4 literals first → hostname last → stick to whichever works."""

    @pytest.mark.asyncio
    async def test_ipv4_literal_tried_before_hostname_and_becomes_sticky(self, monkeypatch):
        calls = []
        behavior = {"api.telegram.org": "timeout", "149.154.167.220": "ok"}
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", _fake_transport_factory(calls, behavior))

        transport = tnet.TelegramFallbackTransport(["149.154.167.220"])
        resp = await transport.handle_async_request(_telegram_request())

        assert resp.status_code == 200
        assert transport._sticky_ip == "149.154.167.220"
        assert [c["url_host"] for c in calls] == ["149.154.167.220"]
        assert calls[0]["host_header"] == "api.telegram.org"
        assert calls[0]["sni_hostname"] == "api.telegram.org"

        # Second request goes straight to sticky IP
        calls.clear()
        resp2 = await transport.handle_async_request(_telegram_request())
        assert resp2.status_code == 200
        assert calls[0]["url_host"] == "149.154.167.220"

    @pytest.mark.asyncio
    async def test_sticky_ip_tried_first_but_falls_through_if_stale(self, monkeypatch):
        """If the sticky IP stops working, the transport retries others."""
        calls = []
        behavior = {
            "api.telegram.org": "timeout",
            "149.154.167.220": "ok",
            "149.154.167.221": "ok",
        }
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", _fake_transport_factory(calls, behavior))

        transport = tnet.TelegramFallbackTransport(["149.154.167.220", "149.154.167.221"])

        # First request: .220 works immediately (IPv4-first) → becomes sticky
        await transport.handle_async_request(_telegram_request())
        assert transport._sticky_ip == "149.154.167.220"

        # Now .220 goes bad too
        calls.clear()
        behavior["149.154.167.220"] = "timeout"

        resp = await transport.handle_async_request(_telegram_request())
        assert resp.status_code == 200
        # Sticky .220 fails → remaining IPv4 .221 works. Hostname is last
        # and is never needed.
        assert [c["url_host"] for c in calls] == ["149.154.167.220", "149.154.167.221"]
        assert transport._sticky_ip == "149.154.167.221"

    @pytest.mark.asyncio
    async def test_hostname_tried_last_when_ipv4_fails(self, monkeypatch):
        """IPv6-only / seed-IP-blocked hosts still reach the hostname last."""
        calls = []
        behavior = {"149.154.167.220": "timeout", "api.telegram.org": "ok"}
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", _fake_transport_factory(calls, behavior))
        transport = tnet.TelegramFallbackTransport(["149.154.167.220"])
        resp = await transport.handle_async_request(_telegram_request())
        assert resp.status_code == 200
        assert [c["url_host"] for c in calls] == ["149.154.167.220", "api.telegram.org"]
        assert transport._sticky_ip is None

class TestFallbackTransportPassthrough:
    """Requests that don't need fallback behavior."""

    @pytest.mark.asyncio
    async def test_non_telegram_host_bypasses_fallback(self, monkeypatch):
        calls = []
        behavior = {}
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", _fake_transport_factory(calls, behavior))

        transport = tnet.TelegramFallbackTransport(["149.154.167.220"])
        request = httpx.Request("GET", "https://example.com/path")
        resp = await transport.handle_async_request(request)

        assert resp.status_code == 200
        assert calls[0]["url_host"] == "example.com"
        assert transport._sticky_ip is tnet._UNSET

class TestFallbackTransportInit:

    def test_uses_proxy_env_for_primary_and_fallback_transports(self, monkeypatch):
        seen_kwargs = []

        def factory(**kwargs):
            seen_kwargs.append(kwargs.copy())
            return FakeTransport([], {})

        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy", "TELEGRAM_PROXY", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", factory)

        transport = tnet.TelegramFallbackTransport(["149.154.167.220"])

        assert transport._fallback_ips == ["149.154.167.220"]
        # Fallback pools are now built lazily (#63311), so __init__ constructs
        # only the primary transport. Force the fallback pool to materialize to
        # observe its kwargs.
        import asyncio

        asyncio.run(transport._get_fallback("149.154.167.220"))
        assert len(seen_kwargs) == 2
        assert all(kwargs["proxy"] == "http://proxy.example:8080" for kwargs in seen_kwargs)

    def test_no_proxy_bypasses_fallback_ip_cidr(self, monkeypatch):
        seen_kwargs = []

        def factory(**kwargs):
            seen_kwargs.append(kwargs.copy())
            return FakeTransport([], {})

        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy", "TELEGRAM_PROXY", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
        monkeypatch.setenv("NO_PROXY", "149.154.160.0/20")
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", factory)

        transport = tnet.TelegramFallbackTransport(["149.154.167.220"])

        assert transport._fallback_ips == ["149.154.167.220"]
        # Lazy fallback build (#63311): materialize the fallback pool.
        import asyncio

        asyncio.run(transport._get_fallback("149.154.167.220"))
        assert len(seen_kwargs) == 2
        assert all("proxy" not in kwargs for kwargs in seen_kwargs)

    def test_forwards_limits_to_inner_transports(self, monkeypatch):
        """Verify that caller-supplied limits reach the inner
        AsyncHTTPTransport instances (#58790).  httpx ignores the
        client-level limits kwarg when a custom transport is
        supplied, so the limits must be forwarded via transport_kwargs.
        """
        seen_kwargs = []

        def factory(**kwargs):
            seen_kwargs.append(kwargs.copy())
            return FakeTransport([], {})

        for key in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy", "TELEGRAM_PROXY", "NO_PROXY", "no_proxy"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", factory)

        custom_limits = httpx.Limits(
            max_connections=42,
            max_keepalive_connections=10,
            keepalive_expiry=30.0,
        )
        transport = tnet.TelegramFallbackTransport(
            ["149.154.167.220"], limits=custom_limits
        )

        # Lazy fallback build (#63311): __init__ builds only the primary; the
        # fallback pool is constructed on demand. Materialize it so both the
        # primary and the fallback are observed.
        import asyncio

        asyncio.run(transport._get_fallback("149.154.167.220"))
        # 1 primary + 1 fallback = 2 AsyncHTTPTransport instances
        assert len(seen_kwargs) == 2
        for kw in seen_kwargs:
            assert "limits" in kw
            # Caller-supplied limits must win over the setdefault default.
            assert kw["limits"] is custom_limits
            assert "socket_options" in kw
            assert any(
                opt[0] == socket.SOL_SOCKET
                and opt[1] == socket.SO_KEEPALIVE
                and opt[2] == 1
                for opt in kw["socket_options"]
            )

class TestFallbackTransportClose:
    @pytest.mark.asyncio
    async def test_aclose_closes_all_transports(self, monkeypatch):
        factory = _fake_transport_factory([], {})
        monkeypatch.setattr(tnet.httpx, "AsyncHTTPTransport", factory)

        transport = tnet.TelegramFallbackTransport(["149.154.167.220", "149.154.167.221"])
        # Lazy fallback build (#63311): materialize both fallback pools so
        # aclose() has something to tear down.
        await transport._get_fallback("149.154.167.220")
        await transport._get_fallback("149.154.167.221")
        await transport.aclose()

        # 1 primary + 2 fallback transports
        assert len(factory.instances) == 3
        assert all(t.closed for t in factory.instances)

# ═══════════════════════════════════════════════════════════════════════════
# Config layer – TELEGRAM_FALLBACK_IPS env → config.extra
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigFallbackIps:
    def test_env_var_populates_config_extra(self, monkeypatch):
        from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides

        monkeypatch.setenv("TELEGRAM_FALLBACK_IPS", "149.154.167.220,149.154.167.221")
        config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok")})
        _apply_env_overrides(config)

        assert config.platforms[Platform.TELEGRAM].extra["fallback_ips"] == [
            "149.154.167.220", "149.154.167.221",
        ]

# ═══════════════════════════════════════════════════════════════════════════
# Adapter layer – _fallback_ips() reads config correctly
# ═══════════════════════════════════════════════════════════════════════════

class TestAdapterFallbackIps:
    def _make_adapter(self, extra=None):
        import sys
        from unittest.mock import MagicMock

        # Ensure telegram mock is in place
        if "telegram" not in sys.modules or not hasattr(sys.modules["telegram"], "__file__"):
            mod = MagicMock()
            mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
            mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
            mod.constants.ChatType.GROUP = "group"
            mod.constants.ChatType.SUPERGROUP = "supergroup"
            mod.constants.ChatType.CHANNEL = "channel"
            mod.constants.ChatType.PRIVATE = "private"
            for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
                sys.modules.setdefault(name, mod)

        from gateway.config import PlatformConfig
        from plugins.platforms.telegram.adapter import TelegramAdapter

        config = PlatformConfig(enabled=True, token="test-token")
        if extra:
            config.extra.update(extra)
        return TelegramAdapter(config)

    def test_list_in_extra(self):
        adapter = self._make_adapter(extra={"fallback_ips": ["149.154.167.220"]})
        assert adapter._fallback_ips() == ["149.154.167.220"]

    def test_csv_string_in_extra(self):
        adapter = self._make_adapter(extra={"fallback_ips": "149.154.167.220,149.154.167.221"})
        assert adapter._fallback_ips() == ["149.154.167.220", "149.154.167.221"]

# ═══════════════════════════════════════════════════════════════════════════
# DoH auto-discovery
# ═══════════════════════════════════════════════════════════════════════════

def _doh_answer(*ips: str) -> dict:
    """Build a minimal DoH JSON response with A records."""
    return {"Answer": [{"type": 1, "data": ip} for ip in ips]}

class FakeDoHClient:
    """Mock httpx.AsyncClient for DoH queries."""

    def __init__(self, responses: dict):
        # responses: URL prefix → (status, json_body) | Exception
        self._responses = responses
        self.requests_made: list[dict] = []

    @staticmethod
    def _make_response(status, body, url):
        """Build an httpx.Response with a request attached (needed for raise_for_status)."""
        request = httpx.Request("GET", url)
        return httpx.Response(status, json=body, request=request)

    async def get(self, url, *, params=None, headers=None, **kwargs):
        self.requests_made.append({"url": url, "params": params, "headers": headers})
        for prefix, action in self._responses.items():
            if url.startswith(prefix):
                if isinstance(action, Exception):
                    raise action
                status, body = action
                return self._make_response(status, body, url)
        return self._make_response(200, {}, url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

class TestSharedSslContext:
    """``shared_ssl_context()`` builds the platform-trust context off-loop, exactly once.

    httpx builds a fresh SSL context (cafile load included) for every client it
    constructs.  With truststore that synchronous construction runs inside the
    OS verifier, so on the event loop it freezes every callback — including the
    connect deadline's own expiry.  The shared context must be built in a worker
    thread, cached, and reused by every caller.
    """

    @pytest.fixture(autouse=True)
    def _reset_cached_context(self, monkeypatch):
        """Fresh cold start per test; every worker started here is joined before the
        monkeypatch restoration so a late build can never write into the next test."""
        from agent import memory_provider

        workers: list = []
        real_spawn = memory_provider.spawn_context_thread

        def _record_worker(*args, **kwargs):
            thread = real_spawn(*args, **kwargs)
            workers.append(thread)
            return thread

        monkeypatch.setattr(memory_provider, "spawn_context_thread", _record_worker)
        monkeypatch.setattr(tnet, "_SSL_CONTEXT", None)
        monkeypatch.setattr(tnet, "_SSL_CONTEXT_FUTURE", None)
        yield
        for worker in workers:
            worker.join(timeout=5.0)
        assert not any(worker.is_alive() for worker in workers), "a shared-context worker outlived its test"

    @pytest.mark.asyncio
    async def test_first_build_is_off_loop_and_loop_keeps_running(self, monkeypatch):
        """The blocking trust-store load runs in a worker; loop timers still fire.

        The factory stands in for the OS verifier stall.  ``threading.Event``s
        synchronise the test with the worker — no sleeps and no tick counting:
        returning from ``await to_thread(started.wait)`` at all proves the loop
        dispatched a callback while *another* thread sat inside the factory.
        """
        import asyncio
        import ssl
        import threading

        real_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        build_started = threading.Event()
        release_build = threading.Event()
        build_threads: list = []

        def _blocking_platform_context():
            build_threads.append(threading.current_thread())
            build_started.set()
            release_build.wait(timeout=5.0)
            return real_ctx

        import agent.ssl_verify as ssl_verify
        monkeypatch.setattr(ssl_verify, "platform_ssl_context", _blocking_platform_context)

        task = asyncio.ensure_future(tnet.shared_ssl_context())
        try:
            started = await asyncio.wait_for(asyncio.to_thread(build_started.wait), timeout=2.0)
            assert started, "the shared context factory never ran"
            assert build_threads and build_threads[0] is not threading.main_thread(), (
                "platform_ssl_context() built on the event-loop thread; a stalled "
                "trust-store load would freeze the gateway"
            )
            # The loop's own timers must still fire while the build is blocked —
            # the connect deadline is a loop timer whose expiry callback must not
            # be starved by context construction.
            timer_fired = asyncio.Event()
            asyncio.get_running_loop().call_soon(timer_fired.set)
            await asyncio.wait_for(timer_fired.wait(), timeout=2.0)
            assert not task.done(), "shared_ssl_context() returned before the factory was released"
            release_build.set()
            ctx = await asyncio.wait_for(task, timeout=2.0)
        finally:
            release_build.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert ctx is real_ctx

    @pytest.mark.asyncio
    async def test_deadline_fires_while_first_build_stalls(self, monkeypatch):
        """The production bound (``agent.deadline``) expires while the build is blocked.

        ``_build_ptb_requests`` and ``discover_fallback_ips`` run under
        ``run_bounded_async``, whose thread timer completes only if the loop can
        process the expiry callback.  With the build off-loop the deadline still
        fires at 2s even though the factory never returned — on the event loop it
        would be starved for the full stall.
        """
        import asyncio
        import ssl
        import threading
        import time

        from agent.deadline import run_bounded_async

        real_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        release_build = threading.Event()

        def _blocking_platform_context():
            release_build.wait(timeout=5.0)
            return real_ctx

        import agent.ssl_verify as ssl_verify
        monkeypatch.setattr(ssl_verify, "platform_ssl_context", _blocking_platform_context)

        started_at = time.monotonic()
        try:
            result = await run_bounded_async(
                tnet.shared_ssl_context(), timeout=2.0, label="shared_ssl_context", dump_on_blocked_loop=False)
        finally:
            release_build.set()

        assert result.timed_out is True, "the 2s production deadline did not fire during a stalled build"
        assert time.monotonic() - started_at >= 2.0, "deadline returned before its own timeout"

        # The deadline abandons the coroutine, not the worker thread: wait for the
        # worker to finish writing the cache while the patch is still installed,
        # so its late write cannot race the fixture teardown.
        wait_until = time.monotonic() + 2.0
        while tnet._SSL_CONTEXT is not real_ctx and time.monotonic() < wait_until:
            await asyncio.to_thread(release_build.wait, 0.01)

    @pytest.mark.asyncio
    async def test_concurrent_first_callers_build_once_and_share_identity(self, monkeypatch):
        """N concurrent cold callers trigger one build; every caller gets it."""
        import asyncio
        import ssl
        import threading

        real_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        build_started = threading.Event()
        release_build = threading.Event()
        build_calls = []

        def _blocking_platform_context():
            build_calls.append(1)
            build_started.set()
            release_build.wait(timeout=5.0)
            return real_ctx

        import agent.ssl_verify as ssl_verify
        monkeypatch.setattr(ssl_verify, "platform_ssl_context", _blocking_platform_context)

        tasks = [asyncio.ensure_future(tnet.shared_ssl_context()) for _ in range(8)]
        try:
            started = await asyncio.wait_for(asyncio.to_thread(build_started.wait), timeout=2.0)
            assert started
            release_build.set()
            contexts = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2.0)
        finally:
            release_build.set()

        assert len(build_calls) == 1, f"cold concurrent callers built the context {len(build_calls)} times"
        assert all(ctx is real_ctx for ctx in contexts)
        # A warm call reuses the cache and never enters the factory again.
        assert await tnet.shared_ssl_context() is real_ctx
        assert len(build_calls) == 1


class TestDiscoverFallbackIps:
    """Tests for discover_fallback_ips() — DoH-based auto-discovery."""

    def _patch_doh(self, monkeypatch, responses, system_dns_ips=None):
        """Wire up fake DoH client and system DNS."""
        client = FakeDoHClient(responses)
        monkeypatch.setattr(tnet.httpx, "AsyncClient", lambda **kw: client)

        if system_dns_ips is not None:
            addrs = [(None, None, None, None, (ip, 443)) for ip in system_dns_ips]
            monkeypatch.setattr(tnet.socket, "getaddrinfo", lambda *a, **kw: addrs)
        else:
            def _fail(*a, **kw):
                raise OSError("dns failed")
            monkeypatch.setattr(tnet.socket, "getaddrinfo", _fail)
        return client

    @pytest.mark.asyncio
    async def test_google_and_cloudflare_ips_collected(self, monkeypatch):
        self._patch_doh(monkeypatch, {
            "https://dns.google": (200, _doh_answer("149.154.167.220")),
            "https://cloudflare-dns.com": (200, _doh_answer("149.154.167.221")),
        }, system_dns_ips=["149.154.166.110"])

        ips = await tnet.discover_fallback_ips()
        assert "149.154.167.220" in ips
        assert "149.154.167.221" in ips

    @pytest.mark.asyncio
    async def test_system_dns_ip_kept_when_doh_confirms(self, monkeypatch):
        """DoH-confirmed IPs are kept even when they match system DNS (#14520).

        The system-DNS IP is often the most reliable path; including it as a
        fallback lets the IP-rewrite retry recover from transient primary-path
        failures instead of jumping straight to the hardcoded seed list.
        """
        self._patch_doh(monkeypatch, {
            "https://dns.google": (200, _doh_answer("149.154.166.110", "149.154.167.220")),
            "https://cloudflare-dns.com": (200, _doh_answer("149.154.166.110")),
        }, system_dns_ips=["149.154.166.110"])

        ips = await tnet.discover_fallback_ips()
        assert ips == ["149.154.166.110", "149.154.167.220"]

    @pytest.mark.asyncio
    async def test_doh_results_deduplicated(self, monkeypatch):
        self._patch_doh(monkeypatch, {
            "https://dns.google": (200, _doh_answer("149.154.167.220")),
            "https://cloudflare-dns.com": (200, _doh_answer("149.154.167.220")),
        }, system_dns_ips=["149.154.166.110"])

        ips = await tnet.discover_fallback_ips()
        assert ips == ["149.154.167.220"]

    @pytest.mark.asyncio
    async def test_all_doh_ips_same_as_system_dns_kept(self, monkeypatch):
        """DoH agrees with system DNS — keep that IP instead of seed list (#14520).

        Previous behavior fell through to ``_SEED_FALLBACK_IPS`` here, but the
        seed addresses are not routable on every network.  When DoH confirms
        the system IP, that IP is the best candidate we have and should be
        used as the fallback target.
        """
        self._patch_doh(monkeypatch, {
            "https://dns.google": (200, _doh_answer("149.154.166.110")),
            "https://cloudflare-dns.com": (200, _doh_answer("149.154.166.110")),
        }, system_dns_ips=["149.154.166.110"])

        ips = await tnet.discover_fallback_ips()
        assert ips == ["149.154.166.110"]

    @pytest.mark.asyncio
    async def test_client_construction_does_not_block_event_loop(self, monkeypatch):
        """Client construction must stay off the event loop (#63309 class).

        httpx builds an SSL context per client; with truststore that load is a
        synchronous OS call that can block for seconds.  On the loop it freezes
        every callback — including the connect deadline's expiry — so reconnect
        attempts time out even though the network path is fine.
        """
        import asyncio
        import threading

        client = FakeDoHClient({"https://dns.google": (200, _doh_answer("149.154.167.220"))})
        construction_started = threading.Event()
        release_construction = threading.Event()
        construction_threads: list = []

        def _stalled_async_client_factory(**kwargs):
            construction_threads.append(threading.current_thread())
            construction_started.set()
            release_construction.wait(timeout=5.0)  # stands in for truststore.load_verify_locations
            return client

        monkeypatch.setattr(tnet.httpx, "AsyncClient", _stalled_async_client_factory)
        monkeypatch.setattr(tnet.socket, "getaddrinfo", lambda *a, **kw: [])
        # The shared context is exercised by TestSharedSslContext; keep this test on
        # the construction seam only.
        monkeypatch.setattr(tnet, "_SSL_CONTEXT", object())

        task = asyncio.ensure_future(tnet.discover_fallback_ips())
        try:
            started = await asyncio.wait_for(asyncio.to_thread(construction_started.wait), timeout=2.0)
            assert started, "AsyncClient construction never started"
            assert construction_threads and construction_threads[0] is not threading.main_thread(), (
                "httpx.AsyncClient was constructed on the event-loop thread; a stalled "
                "trust-store load would freeze the gateway"
            )
            # The loop must still process callbacks while construction is blocked —
            # the production failure froze the connect deadline's own expiry.
            timer_fired = asyncio.Event()
            asyncio.get_running_loop().call_soon(timer_fired.set)
            await asyncio.wait_for(timer_fired.wait(), timeout=2.0)
            assert not task.done(), "discovery finished while the factory was still blocked"
            release_construction.set()
            ips = await asyncio.wait_for(task, timeout=2.0)
        finally:
            release_construction.set()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        assert ips == ["149.154.167.220"]
        assert client.requests_made, "the fake DoH client was never queried"

    @pytest.mark.asyncio
    async def test_hung_system_dns_does_not_gate_doh_results(self, monkeypatch):
        """#63309: socket.getaddrinfo has no timeout of its own — a wedged OS
        resolver must not stall discovery. DoH answers must come back promptly
        even while the system-DNS worker thread is still hanging."""
        import time as _time

        self._patch_doh(monkeypatch, {
            "https://dns.google": (200, _doh_answer("149.154.167.220")),
            "https://cloudflare-dns.com": (200, _doh_answer()),
        }, system_dns_ips=["149.154.166.110"])
        monkeypatch.setattr(tnet, "_DOH_TIMEOUT", 0.2)

        def _hung_getaddrinfo(*a, **kw):
            _time.sleep(0.2)  # far beyond the discovery bound
            raise OSError("resolver wedged")

        monkeypatch.setattr(tnet.socket, "getaddrinfo", _hung_getaddrinfo)

        start = _time.monotonic()
        ips = await tnet.discover_fallback_ips()
        elapsed = _time.monotonic() - start

        assert ips == ["149.154.167.220"]
        assert elapsed < 1.4, f"discovery gated on hung system DNS ({elapsed:.2f}s)"
