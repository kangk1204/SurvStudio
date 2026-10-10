"""Owner-only Tailscale Serve gateway for a separately hosted SurvStudio page.

Run only on loopback. Tailscale Serve strips client-supplied identity headers and
adds the authenticated tailnet identity. The ordinary local app stays unchanged.
"""
from __future__ import annotations

import hmac
import os
from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send, Message


class SiteGateway:
    def __init__(self, application: ASGIApp, *, allowed_login: str, site_origin: str):
        origin = site_origin.strip().rstrip("/")
        parsed = urlsplit(origin)
        if (not allowed_login.strip() or parsed.scheme != "https" or not parsed.hostname
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise ValueError("An owner login and exact HTTPS Site origin are required.")
        self.application = application
        self.allowed_login = allowed_login.strip().lower()
        self.site_origin = origin

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.application(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        headers = Headers(scope=scope)
        logins = headers.getlist("tailscale-user-login")
        client = scope.get("client") or ("", 0)
        if (client[0] not in {"127.0.0.1", "::1"} or len(logins) != 1
                or not hmac.compare_digest(logins[0].strip().lower(), self.allowed_login)):
            await JSONResponse({"detail": "This private research server requires its owner's Tailscale connection."},
                               status_code=403)(scope, receive, send)
            return
        origin = headers.get("origin")
        # Direct visits on the tailnet remain possible; other origins cannot submit requests.
        direct_origin = "https://" + headers.get("host", "")
        if origin is not None and origin not in {self.site_origin, direct_origin}:
            await JSONResponse({"detail": "This page is not an allowed analysis origin."},
                               status_code=403)(scope, receive, send)
            return
        cors = {"Access-Control-Allow-Origin": self.site_origin, "Vary": "Origin"}
        if scope["method"] == "OPTIONS":
            requested = {h.strip().lower() for h in headers.get("access-control-request-headers", "").split(",") if h.strip()}
            if (origin != self.site_origin or headers.get("access-control-request-method") not in {"GET", "HEAD", "POST", "DELETE"}
                    or not requested.issubset({"content-type", "x-requested-with", "x-survstudio-request"})):
                await Response(status_code=403)(scope, receive, send)
                return
            await Response(status_code=204, headers={**cors,
                "Access-Control-Allow-Methods": "GET, HEAD, POST, DELETE",
                "Access-Control-Allow-Headers": "Content-Type, X-Requested-With, X-SurvStudio-Request",
                "Access-Control-Allow-Private-Network": "true", "Access-Control-Max-Age": "600",
            })(scope, receive, send)
            return
        if scope["path"].rstrip("/") == "/api/shutdown":
            await JSONResponse({"detail": "Hosted analyses do not allow shutting down the research server."},
                               status_code=403, headers=cors)(scope, receive, send)
            return
        # Rewrite the local app's origin only after the trusted proxy identity and
        # original browser origin have both been checked. Do not weaken its local guard.
        forwarded = dict(scope)
        forwarded["scheme"] = "http"
        forwarded["headers"] = [(k, v) for k, v in scope["headers"]
                                if k.lower() not in {b"host", b"origin", b"referer"}
                                and not k.lower().startswith(b"x-forwarded-")]
        forwarded["headers"].append((b"host", b"127.0.0.1"))
        if origin is not None:
            forwarded["headers"].append((b"origin", b"http://127.0.0.1"))

        async def send_response(message: Message) -> None:
            if message["type"] == "http.response.start":
                message = dict(message)
                items = [(k, v) for k, v in message.get("headers", [])
                         if not k.lower().startswith(b"access-control-")]
                if origin == self.site_origin:
                    items.extend((k.lower().encode(), v.encode()) for k, v in cors.items())
                items.append((b"x-survstudio-source", os.environ.get("SURVSTUDIO_SITE_SOURCE", "unknown").encode()))
                message["headers"] = items
            await send(message)

        await self.application(forwarded, receive, send_response)


def create_app() -> ASGIApp:
    from .app import app
    return SiteGateway(app, allowed_login=os.environ.get("SURVSTUDIO_SITE_LOGIN", ""),
                       site_origin=os.environ.get("SURVSTUDIO_SITE_ORIGIN", ""))
