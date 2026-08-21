"""OpenAI-compatible streaming reverse proxy with pluggable routing policies.

Env config:
    MANIFOLD_BACKENDS  comma-separated OpenAI-compatible base URLs (required)
    MANIFOLD_POLICY    round_robin | least_outstanding | prefix_aware  (default: least_outstanding)
    MANIFOLD_PORT      default 8000

The prefix key for prefix_aware routing is derived from the stable head of the
conversation: system prompt + first user message. That is what a shared KV-cache
prefix looks like in multi-turn chat traffic.
"""

from __future__ import annotations

import json
import os
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .policies import Policy, make_policy

app = FastAPI(title="manifold-router")

policy: Policy | None = None
client: httpx.AsyncClient | None = None
started = time.time()


@app.on_event("startup")
async def startup() -> None:
    global policy, client
    backends = [u for u in os.environ.get("MANIFOLD_BACKENDS", "").split(",") if u.strip()]
    policy = make_policy(os.environ.get("MANIFOLD_POLICY", "least_outstanding"), backends)
    client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))


@app.on_event("shutdown")
async def shutdown() -> None:
    if client:
        await client.aclose()


def prefix_key_from(body: dict) -> str | None:
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        head = [m for m in messages if m.get("role") == "system"][:1]
        head += [m for m in messages if m.get("role") == "user"][:1]
        return json.dumps([m.get("content") for m in head])
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        return prompt[:256]
    return None


@app.get("/healthz")
async def healthz() -> dict:
    return {"ok": True, "uptime_s": round(time.time() - started, 1)}


@app.get("/stats")
async def stats() -> dict:
    assert policy is not None
    return {
        "policy": policy.name,
        "backends": [
            {"url": b.url, "outstanding": b.outstanding, "total": b.total_requests}
            for b in policy.backends
        ],
    }


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
async def proxy(request: Request) -> Response:
    assert policy is not None and client is not None
    body = await request.json()
    backend = policy.select(prefix_key_from(body))
    url = backend.url + request.url.path
    backend.acquire()

    if body.get("stream"):
        upstream_request = client.build_request("POST", url, json=body)
        upstream = await client.send(upstream_request, stream=True)

        async def relay():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                await upstream.aclose()
                backend.release()

        return StreamingResponse(
            relay(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            headers={"x-manifold-backend": backend.url},
        )

    try:
        upstream = await client.post(url, json=body)
        return JSONResponse(
            content=upstream.json(),
            status_code=upstream.status_code,
            headers={"x-manifold-backend": backend.url},
        )
    finally:
        backend.release()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("MANIFOLD_PORT", "8000")))


if __name__ == "__main__":
    main()
