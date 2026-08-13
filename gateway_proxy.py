"""
AWS API Gateway pass-through proxy transport for httpx (async).

Replicates the request-rewriting behaviour of `requests-ip-rotator`'s
ApiGateway adapter, but for httpx.AsyncClient so the whole codebase stays
async. The `requests-ip-rotator` library itself is still used for the
gateway *lifecycle* (ApiGateway.start()/shutdown() in main.py's lifespan).

How it works (mirrors requests_ip_rotator.ip_rotator.ApiGateway.send):
  1. Picks a random gateway endpoint per request (IP rotation across
     regions + across API Gateway's own egress IP pool).
  2. Rewrites  https://newerp.kluniversity.in/index.php?r=site%2Flogin
            -> https://{api-id}.execute-api.{region}.amazonaws.com/ProxyStage/index.php?r=site%2Flogin
     The gateway's {proxy+} HTTP_PROXY integration forwards path+query to
     the ERP, so the ERP sees normal requests from rotating AWS IPs.
  3. Sets `Host` to the gateway endpoint.
  4. Moves `X-Forwarded-For` -> `X-My-X-Forwarded-For` (randomised if not
     present); the gateway maps it back to `X-Forwarded-For` on the
     integration request, so the ERP never sees the real EC2 IP.
"""

import ipaddress
import logging
import random
from typing import Callable, List

import httpx

logger = logging.getLogger(__name__)

# Stage name created by requests-ip-rotator's ApiGateway.start().
GATEWAY_STAGE = "ProxyStage"

_MAX_IPV4 = 2 ** 32 - 1


class GatewayUnavailableError(Exception):
    """Raised when no API Gateway endpoint is available to route through."""


class ApiGatewayTransport(httpx.AsyncBaseTransport):
    """httpx async transport that routes every request through a random
    AWS API Gateway pass-through endpoint.

    Parameters
    ----------
    endpoints_getter:
        Callable returning the current list of gateway endpoint hosts
        (e.g. ``lambda: gateway_manager.endpoints``). Resolved lazily per
        request so the transport always sees the live gateway state.
    http2:
        Negotiate HTTP/2 with API Gateway (supported on the client-facing
        side). The feedback flow passes False to avoid H2 stream-state
        quirks, matching the previous behaviour.
    verify:
        TLS verification. Terminates at AWS with a valid ACM certificate,
        so verification can (and should) stay enabled — the old
        ``verify=False`` was only needed for the residential MITM proxy.
    """

    def __init__(
        self,
        endpoints_getter: Callable[[], List[str]],
        http2: bool = True,
        verify: bool = True,
        limits: httpx.Limits | None = None,
    ):
        self._endpoints_getter = endpoints_getter
        self._transport = httpx.AsyncHTTPTransport(
            verify=verify,
            http2=http2,
            limits=limits
            or httpx.Limits(
                max_keepalive_connections=50,
                max_connections=200,
                keepalive_expiry=30.0,
            ),
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        endpoints = self._endpoints_getter() or []
        if not endpoints:
            raise GatewayUnavailableError(
                "No API Gateway endpoints available — gateway.start() has not run."
            )

        endpoint = random.choice(endpoints)

        # raw_path keeps percent-encoding intact ("/index.php?r=site%2Flogin").
        site_path = request.url.raw_path.decode("ascii").lstrip("/")
        # Guard against a redirect that was resolved against an already-
        # rewritten gateway URL (would otherwise double the stage prefix).
        if site_path.startswith(f"{GATEWAY_STAGE}/"):
            site_path = site_path[len(GATEWAY_STAGE) + 1:]

        request.url = httpx.URL(f"https://{endpoint}/{GATEWAY_STAGE}/{site_path}")
        request.headers["Host"] = endpoint

        # Auto-generate a random X-Forwarded-For if none was supplied,
        # otherwise AWS forwards the true client IP to the ERP.
        x_forwarded_for = request.headers.get("X-Forwarded-For")
        if x_forwarded_for is None:
            x_forwarded_for = str(
                ipaddress.IPv4Address(random.randint(0, _MAX_IPV4))
            )
        if "X-Forwarded-For" in request.headers:
            del request.headers["X-Forwarded-For"]
        request.headers["X-My-X-Forwarded-For"] = x_forwarded_for

        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._transport.aclose()
