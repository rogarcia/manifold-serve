"""Held requests, class order, preemption and replay, through the real router and sim."""

import asyncio
import json

import httpx
import pytest

from manifold.router.app import Config, create_app
from manifold.router.gate import Caps, RequestClass, Ticket
from manifold.router.policies import RequestCtx
from manifold.router.replicas import ReplicaState
from manifold.sim.app import create_app as create_sim

from .test_integration import SIM_A, SIM_B, Mux, sim_config

CHAT = {"model": "m", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4}
SLOW = {**CHAT, "max_tokens": 200}
BATCH = {"x-manifold-class": "batch"}


class Router:
    def __init__(self, tmp_path, backends=(SIM_A,), caps=None, **config):
        self.sims = {u: create_sim(sim_config(max_num_seqs=64)) for u in (SIM_A, SIM_B)}
        self.mux = Mux(self.sims)
        self.log_path = tmp_path / "d.jsonl"
        config = {"block_tokens": 16, "scrape_ms": 10, "policy": "least_load",
                  "decision_log": str(self.log_path), **config}
        self.app = create_app(Config(backends=list(backends), caps=caps or Caps(), **config),
                              transport=self.mux)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(self.app),
                                        base_url="http://router", timeout=10)

    def post(self, body, headers=None):
        return self.client.post("/v1/chat/completions", json=body, headers=headers)

    def task(self, body, headers=None):
        return asyncio.create_task(self.post(body, headers))

    async def set_state(self, i, state):
        resp = await self.client.post(f"/admin/replicas/{i}/state", json={"state": state})
        assert resp.status_code == 200

    async def demand(self):
        return (await self.client.get("/stats")).json()["demand"]

    async def wait_demand(self, cls, stage, n):
        for _ in range(500):
            if (await self.demand())[cls][stage] == n:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"{cls} {stage} never reached {n}: {await self.demand()}")

    def decisions(self):
        return [json.loads(x) for x in self.log_path.read_text().splitlines()]


@pytest.fixture
async def router(tmp_path):
    made = []

    async def make(**kw):
        r = Router(tmp_path, **kw)
        ctx = r.app.router.lifespan_context(r.app)
        await ctx.__aenter__()
        made.append((r, ctx))
        return r

    yield make
    for r, ctx in made:
        await r.client.aclose()
        await ctx.__aexit__(None, None, None)


HOLD = Caps(hold_interactive_s=5, hold_batch_s=5)


async def test_zero_replicas_with_no_hold_refuses_as_m1_did(router):
    r = await router(initial_state=ReplicaState.OFF)
    resp = await r.post(CHAT)
    assert resp.status_code == 503
    assert resp.headers["retry-after"] == "1"


async def test_first_request_is_held_at_zero_and_served_when_a_slot_is_ready(router):
    r = await router(initial_state=ReplicaState.OFF, caps=HOLD)
    first = r.task(CHAT)
    await r.wait_demand("interactive", "held", 1)
    await r.set_state(0, "ready")
    resp = await first
    assert resp.status_code == 200
    assert resp.headers["x-manifold-backend"] == SIM_A


async def test_a_cancelled_held_request_leaves_no_ghost_in_the_queue(router):
    r = await router(initial_state=ReplicaState.OFF, caps=HOLD)
    t = Ticket(RequestCtx("r", None, [1], [], 1), RequestClass.INTERACTIVE)
    gate = r.app.state.router.gate
    task = asyncio.create_task(gate.acquire(t))
    await r.wait_demand("interactive", "held", 1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert (await r.demand())["interactive"] == {"held": 0, "in_flight": 0}


async def test_past_the_hold_limit_interactive_gets_503_and_batch_429(router):
    r = await router(initial_state=ReplicaState.OFF,
                     caps=Caps(hold_interactive_s=0.05, hold_batch_s=0.05))
    interactive, batch = await asyncio.gather(r.post(CHAT), r.post(CHAT, BATCH))
    assert (interactive.status_code, batch.status_code) == (503, 429)
    assert (await r.demand())["interactive"] == {"held": 0, "in_flight": 0}


async def test_a_replica_at_its_cap_holds_the_next_request(router):
    r = await router(caps=Caps(interactive=1, hold_interactive_s=5))
    first, second = r.task(SLOW), None
    await r.wait_demand("interactive", "in_flight", 1)
    second = r.task(CHAT)
    await r.wait_demand("interactive", "held", 1)
    assert (await first).status_code == 200
    assert (await second).status_code == 200


async def test_batch_waits_while_interactive_is_held(router):
    r = await router(caps=Caps(interactive=1, batch=1, hold_interactive_s=5, hold_batch_s=5))
    first = r.task(SLOW)
    await r.wait_demand("interactive", "in_flight", 1)
    batch = r.task(CHAT, BATCH)
    await r.wait_demand("batch", "held", 1)
    interactive = r.task(CHAT)
    await r.wait_demand("interactive", "held", 1)
    await asyncio.gather(first, interactive, batch)
    order = [d["class"] for d in sorted(r.decisions(), key=lambda d: d["t_granted"])]
    assert order == ["interactive", "interactive", "batch"]


async def test_interactive_preempts_running_batch_and_batch_still_completes(router):
    r = await router(caps=Caps(interactive=1, batch=1, hold_interactive_s=5, hold_batch_s=5))
    batch = r.task(SLOW, BATCH)
    await r.wait_demand("batch", "in_flight", 1)
    interactive = await r.post(CHAT)
    assert interactive.status_code == 200
    resp = await batch
    assert resp.status_code == 200
    [log] = [d for d in r.decisions() if d["class"] == "batch"]
    assert log["preemptions"] == 1 and log["attempts"] == 2
    assert 'manifold_router_replays_total{reason="preempted"} 1.0' in \
        (await r.client.get("/metrics")).text


async def test_a_slot_leaving_service_requeues_its_batch_requests(router):
    r = await router(backends=(SIM_A, SIM_B), caps=HOLD, policy="round_robin")
    batch = r.task(SLOW, BATCH)
    await r.wait_demand("batch", "in_flight", 1)
    await r.set_state(0, "draining")
    resp = await batch
    assert resp.status_code == 200
    assert resp.headers["x-manifold-backend"] == SIM_B


async def test_a_dead_replica_is_retried_before_any_byte_reached_the_client(router):
    r = await router(backends=("http://down", SIM_A), policy="round_robin", max_attempts=2,
                     down_s=5, caps=HOLD)
    resp = await r.post(CHAT)
    assert resp.status_code == 200
    assert resp.headers["x-manifold-backend"] == SIM_A
    stats = (await r.client.get("/stats")).json()
    assert [x["in_flight_requests"] for x in stats["replicas"]] == [0, 0]
    [log] = r.decisions()
    assert log["attempts"] == 2 and log["status"] == 200 and log["error"] is None


async def test_a_stream_that_dies_after_bytes_ends_with_an_error_event(router, monkeypatch):
    class DiesMidStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices": []}\n\n'
            raise httpx.ReadError("gone")

    class Dying(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.path.endswith("/metrics"):
                return httpx.Response(200, text="")
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=DiesMidStream())

    r = await router(max_attempts=3)
    r.app.state.router.client._transport = Dying()
    resp = await r.post({**CHAT, "stream": True})
    assert resp.status_code == 200
    assert '"type": "truncated"' in resp.text
    [log] = r.decisions()
    assert log["attempts"] == 1 and "ReadError" in log["error"]


async def test_bad_class_and_streaming_batch_are_rejected(router):
    r = await router()
    assert (await r.post(CHAT, {"x-manifold-class": "vip"})).status_code == 400
    assert (await r.post({**CHAT, "stream": True}, BATCH)).status_code == 400
