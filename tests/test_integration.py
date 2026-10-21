"""Two fake engines behind the router, in process over ASGI transports."""

import asyncio
import json
import logging
import time

import httpx
import pytest

from manifold.router.app import Config, create_app
from manifold.router.replicas import parse_vllm_metrics
from manifold.sim.app import create_app as create_sim
from manifold.sim.engine import SimConfig

SIM_A, SIM_B = "http://sim-a", "http://sim-b"
SYSTEM = "You are a terse assistant. " * 40
DECISION_FIELDS = {
    "request_id", "session_id", "policy", "chosen", "replica_index", "spilled", "prompt_tokens",
    "total_blocks", "matched_blocks", "scores", "t_received", "t_sent", "t_first_byte", "t_done",
    "status", "error",
}


class Mux(httpx.AsyncBaseTransport):
    """Routes by host to in-process ASGI apps; unknown hosts refuse like a closed port."""

    def __init__(self, apps: dict):
        self.transports = {url: httpx.ASGITransport(app) for url, app in apps.items()}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self.transports.get(f"{request.url.scheme}://{request.url.host}")
        if transport is None:
            raise httpx.ConnectError("connection refused", request=request)
        return await transport.handle_async_request(request)


def sim_config(**kw) -> SimConfig:
    defaults = {"block_tokens": 16, "max_num_seqs": 2, "prefill_us_per_token": 100,
                "decode_ms_per_token": 1}
    return SimConfig(**{**defaults, **kw})


class Harness:
    def __init__(self, tmp_path, backends=(SIM_A, SIM_B), policy="prefix_aware"):
        self.sims = {SIM_A: create_sim(sim_config()), SIM_B: create_sim(sim_config())}
        self.mux = Mux(self.sims)
        self.log_path = tmp_path / "decisions.jsonl"
        self.app = create_app(
            Config(backends=list(backends), policy=policy, block_tokens=16, scrape_ms=10,
                   decision_log=str(self.log_path)),
            transport=self.mux,
        )
        self.router = httpx.AsyncClient(transport=httpx.ASGITransport(self.app),
                                        base_url="http://router")
        self.direct = httpx.AsyncClient(transport=self.mux, timeout=30)

    async def turn(self, session_id: str, messages: list[dict], text: str) -> str:
        messages.append({"role": "user", "content": text})
        resp = await self.router.post(
            "/v1/chat/completions",
            json={"model": "m", "messages": messages, "stream": True, "max_tokens": 8,
                  "conversation_id": session_id},
            headers={"X-Session-ID": session_id},
        )
        assert resp.status_code == 200, resp.text
        reply = "".join(
            json.loads(line[6:])["choices"][0]["delta"].get("content", "")
            for line in resp.text.splitlines()
            if line.startswith("data: {")
        )
        assert resp.text.rstrip().endswith("data: [DONE]")
        messages.append({"role": "assistant", "content": reply})
        return resp.headers["x-manifold-backend"]

    async def set_state(self, index: int, state: str) -> None:
        resp = await self.router.post(f"/admin/replicas/{index}/state", json={"state": state})
        assert resp.status_code == 200, resp.text

    async def stats(self) -> dict:
        return (await self.router.get("/stats")).json()

    def decisions(self) -> list[dict]:
        return [json.loads(line) for line in self.log_path.read_text().splitlines()]

    async def wait_for(self, predicate, timeout=5.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate(await self.stats()):
            assert time.monotonic() < deadline, "condition not reached"
            await asyncio.sleep(0.01)


@pytest.fixture
async def h(tmp_path):
    harness = Harness(tmp_path)
    async with harness.app.router.lifespan_context(harness.app):
        yield harness
    await harness.router.aclose()
    await harness.direct.aclose()


def system_messages() -> list[dict]:
    return [{"role": "system", "content": SYSTEM}]


async def start_session_on_b(h: Harness, session_id: str) -> list[dict]:
    """Idle replicas tie and the tie goes to index 0, so draining A forces turn 1 onto B."""
    messages = system_messages()
    await h.set_state(0, "draining")
    assert await h.turn(session_id, messages, "hello") == SIM_B
    await h.set_state(0, "ready")
    return messages


async def test_prefix_aware_keeps_a_session_on_its_warm_replica(h):
    messages = await start_session_on_b(h, "s1")
    for i in range(4):
        assert await h.turn("s1", messages, f"question {i}") == SIM_B

    later = [d for d in h.decisions() if d["session_id"] == "s1"][1:]
    assert all(d["matched_blocks"][SIM_B] > d["matched_blocks"][SIM_A] for d in later)
    assert all(d["scores"][SIM_B]["prefix"] > 0.9 for d in later)
    assert not any(d["spilled"] for d in later)


async def test_prefix_aware_spills_when_the_warm_replica_is_loaded(h):
    messages = await start_session_on_b(h, "s2")

    # With llm-d's weights a queue gap alone (weight 2) cannot outbid a full prefix match
    # (weight 3), so the hogs fill B's queue and its KV pool.
    async def hog():
        await h.direct.post(f"{SIM_B}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "x" * 20000}], "max_tokens": 1000})

    def b_is_loaded(s: dict) -> bool:
        b = s["replicas"][1]["scraped"] or {}
        return b.get("waiting", 0) >= 2 and b.get("kv_usage", 0.0) >= 0.5

    hogs = [asyncio.create_task(hog()) for _ in range(4)]
    try:
        await h.wait_for(b_is_loaded)
        assert await h.turn("s2", messages, "are you busy?") == SIM_A
    finally:
        for t in hogs:
            t.cancel()
        await asyncio.gather(*hogs, return_exceptions=True)

    last = h.decisions()[-1]
    assert last["chosen"] == SIM_A
    assert last["spilled"] is True
    assert last["scores"][SIM_B]["prefix"] > last["scores"][SIM_A]["prefix"]


async def test_draining_stops_picks_and_forgets_the_index(h):
    messages = await start_session_on_b(h, "s3")
    assert (await h.stats())["replicas"][1]["index_blocks"] > 0

    await h.set_state(1, "draining")
    stats = await h.stats()
    assert stats["replicas"][1]["state"] == "draining"
    assert stats["replicas"][1]["index_blocks"] == 0
    for i in range(3):
        assert await h.turn("s3", messages, f"q{i}") == SIM_A


async def test_bad_admin_state_is_rejected(h):
    assert (await h.router.post("/admin/replicas/0/state", json={"state": "gone"})).status_code \
        == 422
    assert (await h.router.post("/admin/replicas/9/state", json={"state": "ready"})).status_code \
        == 404


async def test_decision_log_lines_carry_every_field(h):
    await h.turn("s4", system_messages(), "hi")
    resp = await h.router.post("/v1/chat/completions",
                               json={"messages": [{"role": "user", "content": "no session"}],
                                     "max_tokens": 2})
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"]

    lines = h.decisions()
    assert len(lines) == 2
    for d in lines:
        assert set(d) == DECISION_FIELDS
        assert d["policy"] == "prefix_aware"
        assert d["status"] == 200 and d["error"] is None
        assert d["t_received"] <= d["t_sent"] <= d["t_first_byte"] <= d["t_done"]
        assert set(d["scores"]) == {SIM_A, SIM_B}
        assert set(d["scores"][d["chosen"]]) == {"prefix", "queue", "kv_util",
                                                 "inflight_tokens", "total"}
        assert d["total_blocks"] == d["prompt_tokens"] // 16
    assert lines[0]["session_id"] == "s4"
    assert lines[1]["session_id"] is None
    assert all(r["in_flight_requests"] == 0 for r in (await h.stats())["replicas"])


async def test_unreachable_replica_returns_502_and_is_logged(tmp_path):
    harness = Harness(tmp_path, backends=("http://down",), policy="round_robin")
    async with harness.app.router.lifespan_context(harness.app):
        resp = await harness.router.post(
            "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x"}]})
        stats = (await harness.router.get("/stats")).json()
    assert resp.status_code == 502
    assert "ConnectError" in resp.json()["error"]["message"]
    [line] = harness.decisions()
    assert line["status"] == 502 and "ConnectError" in line["error"]
    assert line["t_first_byte"] is None
    assert stats["replicas"][0]["in_flight_requests"] == 0
    assert stats["replicas"][0]["in_flight_tokens"] == 0


async def test_router_metrics_and_passthrough(h):
    await h.turn("s5", system_messages(), "hi")
    text = (await h.router.get("/metrics")).text
    assert f'manifold_router_decisions_total{{policy="prefix_aware",replica="{SIM_A}",' \
           'spilled="false"} 1.0' in text
    assert f'manifold_router_inflight_tokens{{replica="{SIM_B}"}} 0.0' in text
    assert "manifold_router_scrape_age_seconds" in text
    assert "manifold_router_overhead_seconds_count 1.0" in text
    models = (await h.router.get("/v1/models")).json()
    assert models["data"][0]["id"] == "sim-model"
    assert (await h.router.get("/version")).status_code == 200


async def test_sim_answers_a_warm_repeat_faster_and_counts_hits():
    sim = create_sim(sim_config(prefill_us_per_token=200))
    body = {"messages": [{"role": "user", "content": "warm me up " * 150}], "max_tokens": 1}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(sim), base_url="http://s") as c:
        async def timed() -> float:
            t0 = time.perf_counter()
            assert (await c.post("/v1/chat/completions", json=body)).status_code == 200
            return time.perf_counter() - t0

        async def hits() -> int:
            return parse_vllm_metrics((await c.get("/metrics")).text).prefix_hits_total

        cold, hits_cold = await timed(), await hits()
        warm, hits_warm = await timed(), await hits()
        assert (await c.post("/reset_prefix_cache")).status_code == 200
        recold = await timed()

    assert cold > 0.3, "1,6xx uncached tokens at 200 us each"
    assert warm < cold / 4
    assert recold > cold / 2
    assert hits_cold == 0
    assert hits_warm >= 1600


async def test_sim_logs_bring_up_lines_in_vllm_wording(caplog):
    sim = create_sim(sim_config(block_tokens=16, kv_blocks=4096))
    with caplog.at_level(logging.INFO, logger="manifold.sim"):
        async with sim.router.lifespan_context(sim):
            pass
    assert "Setting attention block size to 16 tokens" in caplog.messages
    assert "GPU KV cache size: 65,536 tokens" in caplog.messages


def test_config_defaults_bind_loopback_and_require_backends():
    cfg = Config.from_env({"MANIFOLD_BACKENDS": "http://a:1/, http://b:2"})
    assert (cfg.host, cfg.port, cfg.policy) == ("127.0.0.1", 8000, "least_load")
    assert cfg.backends == ["http://a:1", "http://b:2"]
    assert Config.from_env({"MANIFOLD_BACKENDS": "http://a", "MANIFOLD_HOST": "0.0.0.0"}).host \
        == "0.0.0.0"
    with pytest.raises(ValueError, match="MANIFOLD_BACKENDS"):
        Config.from_env({})
