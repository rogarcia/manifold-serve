"""OpenAI-compatible reverse proxy that picks a replica per request with a policy table.

Configuration is environment variables; `Config.from_env` lists them.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from typing import IO, Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel
from starlette.types import Receive, Scope, Send

from .policies import Decision, NoReplicaAvailable, RequestCtx, decide, make_policy
from .prefix import PrefixIndex, block_hashes
from .replicas import Replica, ReplicaState, scrape_forever
from .tokenizer import BytesTokenizer, Tokenizer, make_tokenizer


@dataclass(frozen=True)
class Config:
    backends: list[str]
    policy: str = "least_load"
    host: str = "127.0.0.1"
    port: int = 8000
    tokenizer: str = "bytes"
    chat_template_kwargs: dict[str, Any] = field(default_factory=lambda: {"enable_thinking": False})
    block_tokens: int = 64
    index_blocks: int = 100_000
    spill_factor: float = 2.0
    scrape_ms: int = 100
    default_output_tokens: int = 256
    decision_log: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Config:
        env = dict(os.environ if env is None else env)
        backends = [u.strip().rstrip("/") for u in env.get("MANIFOLD_BACKENDS", "").split(",")]
        backends = [u for u in backends if u]
        if not backends:
            raise ValueError("MANIFOLD_BACKENDS is required (comma-separated base URLs)")
        kwargs = json.loads(env.get("MANIFOLD_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}'))
        if not isinstance(kwargs, dict):
            raise TypeError("MANIFOLD_CHAT_TEMPLATE_KWARGS must be a JSON object")
        cfg = cls(
            backends=backends,
            policy=env.get("MANIFOLD_POLICY", "least_load"),
            host=env.get("MANIFOLD_HOST", "127.0.0.1"),
            port=int(env.get("MANIFOLD_PORT", "8000")),
            tokenizer=env.get("MANIFOLD_TOKENIZER", "bytes"),
            chat_template_kwargs=kwargs,
            block_tokens=int(env.get("MANIFOLD_BLOCK_TOKENS", "64")),
            index_blocks=int(env.get("MANIFOLD_INDEX_BLOCKS", "100000")),
            spill_factor=float(env.get("MANIFOLD_SPILL_FACTOR", "2.0")),
            scrape_ms=int(env.get("MANIFOLD_SCRAPE_MS", "100")),
            default_output_tokens=int(env.get("MANIFOLD_DEFAULT_OUTPUT_TOKENS", "256")),
            decision_log=env.get("MANIFOLD_DECISION_LOG") or None,
        )
        make_policy(cfg.policy)
        if cfg.block_tokens < 1 or cfg.index_blocks < 1 or cfg.scrape_ms < 1:
            raise ValueError("block, index and scrape sizes must be positive")
        return cfg


class _Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.inflight_tokens = Gauge(
            "manifold_router_inflight_tokens", "Router-local in-flight tokens", ["replica"],
            registry=r,
        )
        self.inflight_requests = Gauge(
            "manifold_router_inflight_requests", "Router-local in-flight requests", ["replica"],
            registry=r,
        )
        self.decisions = Counter(
            "manifold_router_decisions", "Routing decisions", ["policy", "replica", "spilled"],
            registry=r,
        )
        self.scrape_age = Gauge(
            "manifold_router_scrape_age_seconds", "Age of the last good /metrics scrape",
            ["replica"], registry=r,
        )
        self.overhead = Histogram(
            "manifold_router_overhead_seconds", "Receive to upstream send", registry=r,
            buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0),
        )

    def render(self, replicas: list[Replica]) -> bytes:
        now = time.time()
        for rep in replicas:
            self.inflight_tokens.labels(rep.url).set(rep.in_flight_tokens)
            self.inflight_requests.labels(rep.url).set(rep.in_flight_requests)
            age = now - rep.scraped.ts if rep.scraped else float("inf")
            self.scrape_age.labels(rep.url).set(age)
        return generate_latest(self.registry)


class _DecisionLog:
    def __init__(self, path: str | None):
        self._f: IO[str] | None = open(path, "a", encoding="utf-8") if path else None  # noqa: SIM115

    def write(self, line: dict[str, Any]) -> None:
        if self._f:
            self._f.write(json.dumps(line) + "\n")
            self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()


class _AdminState(BaseModel):
    state: ReplicaState


def _session_id(request: Request, body: dict) -> str | None:
    sid = request.headers.get("x-session-id") or body.get("conversation_id")
    return None if sid in (None, "") else str(sid)


def _est_output_tokens(body: dict, default: int) -> int:
    for key in ("max_tokens", "max_completion_tokens"):
        if isinstance(body.get(key), int):
            return body[key]
    return default


class _ProxyResponse(Response):
    """Streams the upstream reply through and owns the request's whole lifetime.

    In-flight accounting and the decision-log line live inside this ASGI call, so they are
    released and written on every exit path, including client disconnects.
    """

    def __init__(self, router: _Router, path: str, raw: bytes, ctx: RequestCtx,
                 decision: Decision, t_received: float, auth: str | None):
        self.router, self.path, self.raw, self.ctx = router, path, raw, ctx
        self.decision, self.t_received, self.auth = decision, t_received, auth
        self.background = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        rep = self.decision.chosen
        status, error, started = 502, None, False
        t_sent = t_first = None
        headers = {"content-type": "application/json", "accept-encoding": "identity",
                   "x-request-id": self.ctx.request_id}
        if self.auth:
            headers["authorization"] = self.auth
        ours = [(b"x-manifold-backend", rep.url.encode()),
                (b"x-manifold-request-id", self.ctx.request_id.encode())]
        try:
            with rep.in_flight(self.ctx.load_tokens):
                t_sent = time.time()
                self.router.metrics.overhead.observe(t_sent - self.t_received)
                async with self.router.client.stream(
                    "POST", rep.url + self.path, content=self.raw, headers=headers
                ) as up:
                    ctype = up.headers.get("content-type", "application/json").encode()
                    await send({"type": "http.response.start", "status": up.status_code,
                                "headers": [(b"content-type", ctype), *ours]})
                    started, status = True, up.status_code
                    async for chunk in up.aiter_raw():
                        if t_first is None:
                            t_first = time.time()
                        await send({"type": "http.response.body", "body": chunk,
                                    "more_body": True})
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
        except httpx.HTTPError as e:
            error = f"{type(e).__name__}: {e}"
            if not started:
                body = json.dumps({"error": {"message": f"upstream {rep.url}: {error}",
                                             "type": "upstream_error"}}).encode()
                await send({"type": "http.response.start", "status": 502,
                            "headers": [(b"content-type", b"application/json"), *ours]})
                await send({"type": "http.response.body", "body": body})
            else:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
        except BaseException as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            self.router.log_decision(self.ctx, self.decision, self.t_received, t_sent, t_first,
                                     status, error)


class _Router:
    def __init__(self, config: Config, transport: httpx.AsyncBaseTransport | None):
        self.config = config
        self.transport = transport
        self.replicas = [Replica(url=u, index=i) for i, u in enumerate(config.backends)]
        self.index = PrefixIndex(len(self.replicas), config.index_blocks)
        self.policy = make_policy(config.policy, config.spill_factor)
        self.tokenizer: Tokenizer = make_tokenizer(config.tokenizer)
        self.metrics = _Metrics()
        self.client: httpx.AsyncClient
        self.log: _DecisionLog

    async def tokenize(self, body: dict) -> list[int]:
        tok = self.tokenizer
        if isinstance(body.get("messages"), list):
            kwargs = {**self.config.chat_template_kwargs, **(body.get("chat_template_kwargs") or {})}
            call = (tok.encode_chat, body["messages"], kwargs)
        else:
            prompt = body.get("prompt", "")
            if isinstance(prompt, list) and all(isinstance(t, int) for t in prompt):
                return prompt
            text = prompt if isinstance(prompt, str) else json.dumps(prompt)
            call = (tok.encode_text, text)
        if isinstance(tok, BytesTokenizer):
            return call[0](*call[1:])
        return await asyncio.to_thread(*call)

    def log_decision(self, ctx: RequestCtx, d: Decision, t_received: float,
                     t_sent: float | None, t_first: float | None, status: int,
                     error: str | None) -> None:
        self.log.write({
            "request_id": ctx.request_id,
            "session_id": ctx.session_id,
            "policy": d.policy,
            "chosen": d.chosen.url,
            "replica_index": d.chosen.index,
            "spilled": d.spilled,
            "prompt_tokens": len(ctx.prompt_tokens),
            "total_blocks": len(ctx.block_hashes),
            "matched_blocks": d.matched_blocks,
            "scores": d.scores,
            "t_received": t_received,
            "t_sent": t_sent,
            "t_first_byte": t_first,
            "t_done": time.time(),
            "status": status,
            "error": error,
        })

    async def route(self, request: Request) -> Response:
        t_received = time.time()
        raw = await request.body()
        try:
            body = json.loads(raw)
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return JSONResponse({"error": {"message": "body must be a JSON object"}}, 400)
        tokens = await self.tokenize(body)
        ctx = RequestCtx(
            request_id=uuid.uuid4().hex,
            session_id=_session_id(request, body),
            prompt_tokens=tokens,
            block_hashes=block_hashes(tokens, self.config.block_tokens),
            est_output_tokens=_est_output_tokens(body, self.config.default_output_tokens),
        )
        try:
            decision = decide(self.policy, ctx, self.replicas, self.index)
        except NoReplicaAvailable as e:
            return JSONResponse({"error": {"message": str(e)}}, 503)
        self.index.record(decision.chosen.index, ctx.block_hashes)
        self.metrics.decisions.labels(
            self.policy.name, decision.chosen.url, str(decision.spilled).lower()
        ).inc()
        return _ProxyResponse(self, request.url.path, raw, ctx, decision, t_received,
                              request.headers.get("authorization"))

    async def passthrough(self, path: str) -> Response:
        rep = next((r for r in self.replicas if r.state is ReplicaState.READY), None)
        if rep is None:
            return JSONResponse({"error": {"message": "no ready replica"}}, 503)
        try:
            up = await self.client.get(rep.url + path)
        except httpx.HTTPError as e:
            return JSONResponse({"error": {"message": f"upstream {rep.url}: {e!r}"}}, 502)
        return Response(up.content, up.status_code,
                        media_type=up.headers.get("content-type", "application/json"))

    def set_state(self, index: int, state: ReplicaState) -> dict:
        if not 0 <= index < len(self.replicas):
            raise HTTPException(404, f"no replica {index}")
        rep = self.replicas[index]
        rep.state = state
        if state is not ReplicaState.READY:
            self.index.forget(index)
        return {"index": index, "url": rep.url, "state": state.value}

    def stats(self) -> dict:
        return {
            "policy": self.policy.name,
            "replicas": [
                {
                    "index": r.index,
                    "url": r.url,
                    "state": r.state.value,
                    "in_flight_tokens": r.in_flight_tokens,
                    "in_flight_requests": r.in_flight_requests,
                    "index_blocks": self.index.size(r.index),
                    "scraped": asdict(r.scraped) if r.scraped else None,
                }
                for r in self.replicas
            ],
        }


def create_app(config: Config, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    router = _Router(config, transport)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        router.client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(600.0, connect=10.0, pool=None),
            limits=httpx.Limits(max_connections=4096, max_keepalive_connections=1024),
        )
        router.log = _DecisionLog(config.decision_log)
        tasks = [asyncio.create_task(scrape_forever(router.client, r, config.scrape_ms / 1000))
                 for r in router.replicas]
        try:
            yield
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await router.client.aclose()
            router.log.close()

    app = FastAPI(title="manifold-router", lifespan=lifespan)
    app.state.router = router
    started = time.time()

    app.add_api_route("/v1/chat/completions", router.route, methods=["POST"])
    app.add_api_route("/v1/completions", router.route, methods=["POST"])

    @app.get("/v1/models")
    async def models() -> Response:
        return await router.passthrough("/v1/models")

    @app.get("/version")
    async def version() -> Response:
        return await router.passthrough("/version")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True, "uptime_s": round(time.time() - started, 1)}

    @app.get("/stats")
    async def stats() -> dict:
        return router.stats()

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(router.metrics.render(router.replicas), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/replicas/{index}/state")
    async def admin_state(index: int, body: _AdminState) -> dict:
        return router.set_state(index, body.state)

    return app


def main() -> None:
    import uvicorn

    try:
        config = Config.from_env()
    except (ValueError, TypeError) as e:
        raise SystemExit(f"manifold-router: {e}") from None
    uvicorn.run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
