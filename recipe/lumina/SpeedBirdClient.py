#!/usr/bin/env python3
"""Lightweight client for issuing SpeedBird MCP search requests."""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any, Dict, Optional, Sequence

import aiohttp

from recipe.lumina.cert_kv_helper import acquire_token_from_kv_cert

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class _NoOpRateLimiter:
    """Placeholder limiter; swap out with a real limiter when available."""

    async def __aenter__(self) -> "_NoOpRateLimiter":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> Optional[bool]:
        return False


def get_speedbird_rate_limiter() -> _NoOpRateLimiter:
    """Return a placeholder rate limiter so the call sites stay intact."""

    return _NoOpRateLimiter()


class SpeedbirdMcpClient:
    """Minimal SpeedBird MCP client supporting only the search tool."""

    def __init__(
        self,
        base_url: str,
        mcp_path: str = "/speedbird/mcp",
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        scope: Optional[str] = None,
        keyvault_url: Optional[str] = None,
        certificate_name: Optional[str] = None,
        use_auth: bool = True,
        protocol_version: str = "2025-06-18",
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.mcp_path = mcp_path
        self.protocol_version = protocol_version
        self.use_auth = use_auth

        self.tenant_id = tenant_id
        self.client_id = client_id
        self.scope = scope
        self.keyvault_url = keyvault_url
        self.certificate_name = certificate_name

        self._session = session
        self._token_cache: Optional[tuple[str, float]] = None

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _build_url(self) -> str:
        return f"{self.base_url}{self.mcp_path}"

    def _build_headers(self, token: Optional[str]) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Debug-Mode": "false",
            "X-Protocol-Version": self.protocol_version,
            "structuredoutput:": "structured",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _auth_configured(self) -> bool:
        return bool(
            self.tenant_id
            and self.client_id
            and self.scope
            and self.keyvault_url
            and self.certificate_name
        )

    async def _get_token(self) -> Optional[str]:
        if not self.use_auth:
            return None

        if not self._auth_configured():
            logger.warning("SpeedBird auth disabled because credentials are missing.")
            return None

        now = time.time()
        if self._token_cache and self._token_cache[1] - 300 > now:
            return self._token_cache[0]

        token_response = acquire_token_from_kv_cert(
            keyvault_url=self.keyvault_url,
            cert_name=self.certificate_name,
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            scope=self.scope,
        )
        token = token_response.get("access_token")
        expires_in = token_response.get("expires_in") or 3600
        self._token_cache = (token, now + expires_in)
        return token

    async def search(
        self,
        query: str,
        max_results: int = 10,
        language: str = "en",
        region: str = "us",
        domains: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Call the SpeedBird MCP search tool and return the raw response."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        max_results = max(1, min(int(max_results), 20))

        arguments: Dict[str, Any] = {
            "query": query[:1000],
            "maxResults": max_results,
            "language": language or "en",
            "region": region or "us",
        }
        if domains:
            filtered_domains = [str(domain) for domain in domains if domain]
            if filtered_domains:
                arguments["domains"] = filtered_domains

        payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex[:8],
            "method": "tools/call",
            "params": {
                "name": "search",
                "arguments": arguments,
            },
        }

        url = self._build_url()
        token = await self._get_token()
        headers = self._build_headers(token)

        async with get_speedbird_rate_limiter():
            try:
                session = await self._get_session()
                async with session.post(url, json=payload, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()
                    logger.info(
                        "SpeedBird search success status=%s query_preview=%s",
                        response.status,
                        query[:50],
                    )
                    return data
            except aiohttp.ClientError as exc:
                logger.error("SpeedBird search client error: %s", exc)
                raise
            except Exception as exc:  # pragma: no cover - defensive log
                logger.error("SpeedBird search failed: %s", exc)
                raise
