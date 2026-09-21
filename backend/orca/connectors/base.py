"""Shared HTTP plumbing for every upstream connector.

Design rules, all of which exist because the upstream services are flaky:

* one shared `httpx.AsyncClient` with a browser-ish UA (several .gov.in hosts
  sit behind a WAF that rejects default client UAs);
* TTL memory cache plus an on-disk cache for large, slow-moving artefacts such
  as the EEZ polygon and THREDDS coordinate arrays;
* a connector never raises into the agent layer - it returns `None`/empty and
  records the failure so the agent can degrade and say so.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from ..config import get_settings

log = logging.getLogger("orca.connectors")


@dataclass
class FetchOutcome:
    """Result of one upstream call, success or not."""

    ok: bool
    url: str
    status: int = 0
    text: str = ""
    json_body: Any = None
    elapsed_ms: int = 0
    error: str = ""
    from_cache: bool = False


@dataclass
class SourceHealth:
    """Rolling per-source health so the UI can show what degraded."""

    name: str
    ok_count: int = 0
    fail_count: int = 0
    last_error: str = ""
    last_ok_ms: int = 0

    @property
    def degraded(self) -> bool:
        return self.fail_count > 0 and self.ok_count == 0


class _MemoryCache:
    def __init__(self) -> None:
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            hit = self._data.get(key)
            if not hit:
                return None
            expires_at, value = hit
            if expires_at < time.time():
                self._data.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: Any, ttl: float) -> None:
        async with self._lock:
            self._data[key] = (time.time() + ttl, value)


_MEM = _MemoryCache()


class HttpConnector:
    """Base class: subclasses get `self.get_text`, `self.get_json`, health."""

    #: short name used in health reporting and trace output
    source_name = "http"

    _client: httpx.AsyncClient | None = None
    _client_lock = asyncio.Lock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self.health = SourceHealth(name=self.source_name)

    # -- client ------------------------------------------------------------- #

    @classmethod
    async def client(cls) -> httpx.AsyncClient:
        if HttpConnector._client is None:
            async with HttpConnector._client_lock:
                if HttpConnector._client is None:
                    s = get_settings()
                    HttpConnector._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(s.http_timeout_s),
                        follow_redirects=True,
                        headers={
                            # Some .gov.in WAFs 403 non-browser UAs. We identify
                            # ORCA in the Accept chain instead of spoofing wholly.
                            "User-Agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/126.0.0.0 Safari/537.36 " + s.user_agent
                            ),
                            "Accept-Language": "en-IN,en;q=0.9",
                        },
                        limits=httpx.Limits(
                            max_connections=16, max_keepalive_connections=8
                        ),
                    )
        return HttpConnector._client

    @classmethod
    async def aclose(cls) -> None:
        if HttpConnector._client is not None:
            await HttpConnector._client.aclose()
            HttpConnector._client = None

    # -- disk cache --------------------------------------------------------- #

    def _disk_path(self, key: str, suffix: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()[:20]
        root = Path(self.settings.cache_dir) / self.source_name
        root.mkdir(parents=True, exist_ok=True)
        return root / f"{digest}{suffix}"

    def disk_get(self, key: str, suffix: str, max_age_s: float) -> str | None:
        p = self._disk_path(key, suffix)
        if p.exists() and (time.time() - p.stat().st_mtime) < max_age_s:
            try:
                return p.read_text(encoding="utf-8")
            except OSError:
                return None
        return None

    def disk_put(self, key: str, suffix: str, payload: str) -> None:
        try:
            self._disk_path(key, suffix).write_text(payload, encoding="utf-8")
        except OSError as exc:  # pragma: no cover - disk issues are non-fatal
            log.warning("disk cache write failed: %s", exc)

    # -- fetch -------------------------------------------------------------- #

    async def fetch(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        ttl: float | None = None,
        as_json: bool = False,
        retries: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> FetchOutcome:
        ttl = self.settings.cache_ttl_s if ttl is None else ttl
        retries = self.settings.http_retries if retries is None else retries
        cache_key = f"{url}?{json.dumps(params, sort_keys=True, default=str)}"

        if ttl > 0:
            cached = await _MEM.get(cache_key)
            if cached is not None:
                out: FetchOutcome = cached
                return FetchOutcome(
                    ok=out.ok,
                    url=out.url,
                    status=out.status,
                    text=out.text,
                    json_body=out.json_body,
                    elapsed_ms=out.elapsed_ms,
                    from_cache=True,
                )

        client = await self.client()
        last_error = ""
        for attempt in range(retries + 1):
            started = time.perf_counter()
            try:
                resp = await client.get(url, params=params, headers=headers)
                elapsed = int((time.perf_counter() - started) * 1000)
                if resp.status_code >= 400:
                    last_error = f"HTTP {resp.status_code}"
                    # 4xx will not fix itself on retry; 5xx might
                    if resp.status_code < 500:
                        break
                else:
                    body = None
                    if as_json:
                        try:
                            body = resp.json()
                        except ValueError as exc:
                            last_error = f"bad json: {exc}"
                            break
                    outcome = FetchOutcome(
                        ok=True,
                        url=str(resp.url),
                        status=resp.status_code,
                        text=resp.text if not as_json else "",
                        json_body=body,
                        elapsed_ms=elapsed,
                    )
                    self.health.ok_count += 1
                    self.health.last_ok_ms = elapsed
                    if ttl > 0:
                        await _MEM.set(cache_key, outcome, ttl)
                    return outcome
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                await asyncio.sleep(0.6 * (attempt + 1))

        self.health.fail_count += 1
        self.health.last_error = last_error
        log.warning("%s fetch failed %s: %s", self.source_name, url, last_error)
        return FetchOutcome(ok=False, url=url, error=last_error)

    async def get_text(self, url: str, **kw: Any) -> str | None:
        out = await self.fetch(url, **kw)
        return out.text if out.ok else None

    async def get_json(self, url: str, **kw: Any) -> Any | None:
        kw["as_json"] = True
        out = await self.fetch(url, **kw)
        return out.json_body if out.ok else None


@dataclass
class ConnectorRegistry:
    """Keeps one instance of each connector and aggregates health."""

    instances: dict[str, HttpConnector] = field(default_factory=dict)

    def add(self, connector: HttpConnector) -> HttpConnector:
        self.instances[connector.source_name] = connector
        return connector

    def degraded(self) -> list[str]:
        return [
            f"{c.source_name} ({c.health.last_error})"
            for c in self.instances.values()
            if c.health.fail_count and c.health.last_error
        ]
