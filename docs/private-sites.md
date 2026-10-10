# Private Sites workspace

The hosted page serves the same SurvStudio interface and calls an owner-only research server over Tailscale. Uploads are processed on that server, so the hosted page states this explicitly. The local installation continues to use its own same-origin API.

Serve the gateway on loopback only:

```
uvicorn survival_toolkit.site_gateway:create_app --factory --host 127.0.0.1 --port 8970 --no-proxy-headers
```

Set `SURVSTUDIO_SITE_LOGIN` to the owner's Tailscale login and `SURVSTUDIO_SITE_ORIGIN` to the exact private Sites origin. Set `SURVSTUDIO_SITE_SOURCE` to the deployed source SHA. The gateway rejects missing, duplicate, or different login headers; a non-loopback client; unexpected browser origins; and hosted shutdown requests. It leaves the original app's local request guard in place. CORS is limited to the declared Site origin. The static frontend uses an HTTPS API origin embedded by the build, with no credentials in browser code.

[Tailscale Serve](https://tailscale.com/docs/features/tailscale-serve) strips incoming identity headers before adding the authenticated identity. Use Serve, and bind the application to loopback; direct network binding would invalidate that trust boundary. Public Funnel is outside this deployment arrangement.

A disconnected browser receives instructions to reconnect Tailscale. The Site and server permissions remain owner-only. This deployment is a private research workspace; it does not establish statistical qualification or clinical suitability.
