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
from enum import Enum
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

from .gate import Caps, Gate, RequestClass, Shed, Ticket
from .policies import Decision, RequestCtx, decide, make_policy
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
    initial_state: ReplicaState = ReplicaState.READY
    caps: Caps = field(default_factory=Caps)
    max_attempts: int = 1
    down_s: float = 0.0

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
            initial_state=ReplicaState(env.get("MANIFOLD_INITIAL_STATE", "ready")),
            caps=Caps(
                interactive=float(env.get("MANIFOLD_CAP_INTERACTIVE", "inf")),
                batch=float(env.get("MANIFOLD_CAP_BATCH", "inf")),
                hold_interactive_s=float(env.get("MANIFOLD_HOLD_INTERACTIVE_S", "0")),
                hold_batch_s=float(env.get("MANIFOLD_HOLD_BATCH_S", "0")),
            ),
            max_attempts=int(env.get("MANIFOLD_MAX_ATTEMPTS", "1")),
            down_s=float(env.get("MANIFOLD_DOWN_S", "0")),
        )
        make_policy(cfg.policy)
        if min(cfg.block_tokens, cfg.index_blocks, cfg.scrape_ms, cfg.max_attempts) < 1:
            raise ValueError("block, index, scrape and attempt counts must be positive")
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
        self.demand = Gauge(
            "manifold_router_demand_requests", "Requests held or in flight, per class",
            ["class", "stage"], registry=r,
        )
        self.preemptions = Gauge(
            "manifold_router_preemptions", "Batch requests preempted since start", registry=r,
        )
        self.replays = Counter(
            "manifold_router_replays", "Requests sent again after no byte reached the client",
            ["reason"], registry=r,
        )
        self.shed = Counter(
            "manifold_router_shed", "Requests refused past the hold limit", ["class"], registry=r,
        )

    def render(self, replicas: list[Replica], gate: Gate) -> bytes:
        now = time.time()
        for cls, (held, in_flight) in gate.demand().items():
            self.demand.labels(cls.value, "held").set(held)
            self.demand.labels(cls.value, "in_flight").set(in_flight)
        self.preemptions.set(gate.preemptions)
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


def _error_body(message: str, kind: str) -> bytes:
    return json.dumps({"error": {"message": message, "type": kind}}).encode()


class _Outcome(Enum):
    DONE = "done"
    PREEMPTED = "preempted"
    FAILED = "failed"
    REFUSED = "refused"


class _ProxyResponse(Response):
    """Owns the request's whole lifetime: admission, attempts, streaming, accounting, logging.

    The replay rule covers preemption and failure alike: a request is sent again if and only if
    no byte reached the client. Everything is released on every exit path, including client
    disconnects, because it all lives inside this ASGI call.
    """

    def __init__(self, router: _Router, path: str, raw: bytes, ticket: Ticket, t_received: float,
                 auth: str | None):
        self.router, self.path, self.raw, self.ticket = router, path, raw, ticket
        self.t_received, self.auth = t_received, auth
        self.background = None
        self.status, self.error = 502, None
        self.t_granted = self.t_sent = self.t_first = None
        self.attempts = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        router, t, decision = self.router, self.ticket, None
        failures, replay = 0, False
        try:
            while True:
                try:
                    decision = await router.gate.acquire(t, front=replay)
                except Shed as e:
                    self.status, self.error = e.status, str(e)
                    router.metrics.shed.labels(t.cls.value).inc()
                    await _send_error(send, e.status, _error_body(str(e), "overloaded"),
                                      [(b"retry-after", str(e.retry_after_s).encode())])
                    return
                self.t_granted = time.time()
                self.attempts += 1
                outcome = await self._attempt(decision, send)
                router.gate.release(t)
                if outcome is _Outcome.DONE:
                    return
                if outcome in (_Outcome.FAILED, _Outcome.REFUSED):
                    failures += 1
                    if outcome is _Outcome.FAILED:
                        router.mark_down(decision.chosen)
                    if t.started or failures >= router.config.max_attempts:
                        await self._finish_failure(decision, send)
                        return
                router.metrics.replays.labels(outcome.value).inc()
                replay = True
        except BaseException as e:
            self.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            router.gate.release(t)
            if decision is not None:
                router.log_decision(self, decision)

    async def _attempt(self, decision: Decision, send: Send) -> _Outcome:
        t = self.ticket
        t.task = asyncio.create_task(self._forward(decision.chosen, send))
        try:
            return await t.task
        except asyncio.CancelledError:
            if t.preempted:
                return _Outcome.PREEMPTED
            t.task.cancel()
            raise
        finally:
            t.task = None

    async def _forward(self, rep: Replica, send: Send) -> _Outcome:
        t = self.ticket
        headers = {"content-type": "application/json", "accept-encoding": "identity",
                   "x-request-id": t.ctx.request_id}
        if self.auth:
            headers["authorization"] = self.auth
        self.t_sent = time.time()
        self.router.metrics.overhead.observe(self.t_sent - self.t_granted)
        try:
            async with self.router.client.stream(
                "POST", rep.url + self.path, content=self.raw, headers=headers
            ) as up:
                if up.status_code >= 500:  # vLLM refuses new work this way while it shuts down.
                    # Not marked down: one request that crashes the engine must not take a replica out.
                    body = (await up.aread())[:200].decode(errors="replace")
                    self.error = f"HTTP {up.status_code}: {body}"
                    return _Outcome.REFUSED
                ctype = up.headers.get("content-type", "application/json").encode()
                t.started = True
                await send({"type": "http.response.start", "status": up.status_code,
                            "headers": [(b"content-type", ctype), *self._ours(rep)]})
                self.status = up.status_code
                async for chunk in up.aiter_raw():
                    if self.t_first is None:
                        self.t_first = time.time()
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
        except httpx.HTTPError as e:
            self.error = f"{type(e).__name__}: {e}"
            return _Outcome.FAILED
        self.error = None
        return _Outcome.DONE

    def _ours(self, rep: Replica) -> list[tuple[bytes, bytes]]:
        return [(b"x-manifold-backend", rep.url.encode()),
                (b"x-manifold-request-id", self.ticket.ctx.request_id.encode())]

    async def _finish_failure(self, decision: Decision, send: Send) -> None:
        if not self.ticket.started:
            msg = f"upstream {decision.chosen.url}: {self.error}"
            await _send_error(send, 502, _error_body(msg, "upstream_error"),
                              self._ours(decision.chosen))
            return
        # The client already holds a 200 stream, so the failure has to travel inside it.
        event = b"data: " + _error_body(f"upstream lost: {self.error}", "truncated") + b"\n\n"
        await send({"type": "http.response.body", "body": event, "more_body": True})
        await send({"type": "http.response.body", "body": b"", "more_body": False})


async def _send_error(send: Send, status: int, body: bytes,
                      headers: list[tuple[bytes, bytes]]) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), *headers]})
    await send({"type": "http.response.body", "body": body})


class _Router:
    def __init__(self, config: Config, transport: httpx.AsyncBaseTransport | None):
        self.config = config
        self.transport = transport
        self.replicas = [Replica(url=u, index=i) for i, u in enumerate(config.backends)]
        self.index = PrefixIndex(len(self.replicas), config.index_blocks)
        self.policy = make_policy(config.policy, config.spill_factor)
        self.tokenizer: Tokenizer = make_tokenizer(config.tokenizer)
        self.metrics = _Metrics()
        for r in self.replicas:
            r.state = config.initial_state
        self.gate = Gate(self.replicas, config.caps, self.pick)
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

    def pick(self, ctx: RequestCtx, cands: list[Replica]) -> Decision:
        decision = decide(self.policy, ctx, cands, self.index)
        self.index.record(decision.chosen.index, ctx.block_hashes)
        self.metrics.decisions.labels(
            self.policy.name, decision.chosen.url, str(decision.spilled).lower()
        ).inc()
        return decision

    def mark_down(self, rep: Replica) -> None:
        if self.config.down_s > 0:
            rep.down_until = time.monotonic() + self.config.down_s
            asyncio.get_running_loop().call_later(self.config.down_s, self.gate.pump)

    def log_decision(self, r: _ProxyResponse, d: Decision) -> None:
        ctx = r.ticket.ctx
        self.log.write({
            "request_id": ctx.request_id,
            "session_id": ctx.session_id,
            "class": r.ticket.cls.value,
            "policy": d.policy,
            "chosen": d.chosen.url,
            "replica_index": d.chosen.index,
            "spilled": d.spilled,
            "prompt_tokens": len(ctx.prompt_tokens),
            "total_blocks": len(ctx.block_hashes),
            "matched_blocks": d.matched_blocks,
            "scores": d.scores,
            "attempts": r.attempts,
            "preemptions": r.ticket.preemptions,
            "t_received": r.t_received,
            "t_granted": r.t_granted,
            "t_sent": r.t_sent,
            "t_first_byte": r.t_first,
            "t_done": time.time(),
            "status": r.status,
            "error": r.error,
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
        try:
            cls = RequestClass(request.headers.get("x-manifold-class", "interactive"))
        except ValueError:
            return JSONResponse({"error": {"message": "unknown x-manifold-class"}}, 400)
        if cls is RequestClass.BATCH and body.get("stream"):
            return JSONResponse({"error": {"message": "batch requests must not stream"}}, 400)
        tokens = await self.tokenize(body)
        ctx = RequestCtx(
            request_id=uuid.uuid4().hex,
            session_id=_session_id(request, body),
            prompt_tokens=tokens,
            block_hashes=block_hashes(tokens, self.config.block_tokens),
            est_output_tokens=_est_output_tokens(body, self.config.default_output_tokens),
        )
        return _ProxyResponse(self, request.url.path, raw, Ticket(ctx, cls), t_received,
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
            self.gate.evict(rep)
        if state is ReplicaState.OFF:
            rep.scraped = None
        self.gate.pump()
        return {"index": index, "url": rep.url, "state": state.value}

    def stats(self) -> dict:
        return {
            "policy": self.policy.name,
            "demand": {c.value: {"held": h, "in_flight": f}
                       for c, (h, f) in self.gate.demand().items()},
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
        return Response(router.metrics.render(router.replicas, router.gate), media_type=CONTENT_TYPE_LATEST)

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
