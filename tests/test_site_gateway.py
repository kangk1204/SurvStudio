from __future__ import annotations

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.testclient import TestClient

from survival_toolkit.site_gateway import SiteGateway

ORIGIN = "https://survstudio.example.test"
LOGIN = "owner@example.test"


async def echo(scope, receive, send):
    request = Request(scope, receive)
    await JSONResponse({"host": request.headers.get("host"), "origin": request.headers.get("origin"),
                        "method": scope["method"]})(scope, receive, send)


def gateway_client(client_host="127.0.0.1"):
    gateway = SiteGateway(echo, allowed_login=LOGIN, site_origin=ORIGIN)
    return TestClient(gateway, client=(client_host, 12345), base_url="https://server.example.test")


@pytest.mark.parametrize("login", [None, "", "other@example.test"])
def test_identity_is_required_before_forwarding(login):
    headers = {} if login is None else {"Tailscale-User-Login": login}
    assert gateway_client().get("/api/health", headers=headers).status_code == 403


def test_direct_network_cannot_forge_proxy_identity():
    assert gateway_client("100.64.1.2").get("/api/health", headers={"Tailscale-User-Login": LOGIN}).status_code == 403


def test_duplicate_identity_is_rejected():
    assert gateway_client().get("/api/health", headers=[("Tailscale-User-Login", LOGIN)] * 2).status_code == 403


def test_owner_requests_forward_only_after_origin_check():
    headers = {"Tailscale-User-Login": LOGIN, "Origin": ORIGIN}
    response = gateway_client().post("/api/kaplan-meier", headers=headers, json={})
    assert response.status_code == 200
    assert response.json() == {"host": "127.0.0.1", "origin": "http://127.0.0.1", "method": "POST"}
    assert response.headers["access-control-allow-origin"] == ORIGIN
    headers["Origin"] = "https://other.example.test"
    assert gateway_client().post("/api/kaplan-meier", headers=headers, json={}).status_code == 403


def test_preflight_requires_the_owner_and_exact_origin():
    headers = {"Origin": ORIGIN, "Access-Control-Request-Method": "POST",
               "Access-Control-Request-Headers": "content-type"}
    client = gateway_client()
    assert client.options("/api/upload", headers=headers).status_code == 403
    headers["Tailscale-User-Login"] = LOGIN
    response = client.options("/api/upload", headers=headers)
    assert response.status_code == 204
    assert response.headers["access-control-allow-private-network"] == "true"
    headers["Access-Control-Request-Headers"] = "authorization"
    assert client.options("/api/upload", headers=headers).status_code == 403


def test_hosted_shutdown_is_never_forwarded():
    assert gateway_client().post("/api/shutdown", headers={"Tailscale-User-Login": LOGIN, "Origin": ORIGIN}).status_code == 403


@pytest.mark.parametrize("origin", ["http://example.test", "https://example.test/path", "https://user@example.test", "https://example.test?query"])
def test_unsafe_configuration_fails_closed(origin):
    with pytest.raises(ValueError):
        SiteGateway(echo, allowed_login=LOGIN, site_origin=origin)
