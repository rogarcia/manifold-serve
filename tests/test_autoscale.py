"""The whole loop as real processes: router, autoscaler, two slow-starting sims."""

import asyncio
import json
import os
import signal
import socket
import subprocess
import sys

import httpx

ENV = {k: v for k, v in os.environ.items() if not k.startswith("MANIFOLD")}


def free_ports(n):
    socks = [socket.socket() for _ in range(n)]
    for s in socks:
        s.bind(("127.0.0.1", 0))
    ports = [s.getsockname()[1] for s in socks]
    for s in socks:
        s.close()
    return ports


def spawn(module, env):
    return subprocess.Popen([sys.executable, "-m", module], env=env)


async def wait(predicate, timeout, what):
    for _ in range(int(timeout / 0.05)):
        got = await predicate()
        if got:
            return got
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


async def test_zero_up_to_two_down_to_zero_and_crash_recovery(tmp_path):
    router_port, a, b = free_ports(3)
    urls = [f"http://127.0.0.1:{a}", f"http://127.0.0.1:{b}"]
    ports = json.dumps([a, b])
    engine = (f"{sys.executable} -c \"import os, sys; slot = int(sys.argv[1]); "
              f"port = {ports}[slot]; os.execv(sys.executable, [sys.executable, '-m', "
              "'manifold.sim', '--port', str(port), '--start-delay-s', '0.3', "
              "'--max-num-seqs', '512', '--decode-ms-per-token', '5', "
              "'--shutdown-timeout-s', '2'])\" {slot}")
    env = {**ENV, "MANIFOLD_BACKENDS": ",".join(urls), "MANIFOLD_PORT": str(router_port),
           "MANIFOLD_INITIAL_STATE": "off", "MANIFOLD_CAP_INTERACTIVE": "96",
           "MANIFOLD_CAP_BATCH": "64", "MANIFOLD_HOLD_INTERACTIVE_S": "30",
           "MANIFOLD_HOLD_BATCH_S": "30", "MANIFOLD_MAX_ATTEMPTS": "3", "MANIFOLD_DOWN_S": "1",
           "MANIFOLD_POLICY": "least_load", "MANIFOLD_ROUTER_URL": f"http://127.0.0.1:{router_port}",
           "MANIFOLD_ENGINE_CMD": engine, "MANIFOLD_RUN_DIR": str(tmp_path / "run"),
           "MANIFOLD_TICK_LOG": str(tmp_path / "ticks.jsonl"),
           "MANIFOLD_SPEC": json.dumps({
               "tick_s": 0.1, "target_interactive": 4, "up_window_s": 0.2, "settle_s": 0.2,
               "down_window_s": 0.5, "idle_to_zero_s": 1.0, "drain_deadline_s": 3})}
    procs = {"router": spawn("manifold.router", env), "auto": spawn("manifold.autoscaler", env)}
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{router_port}", timeout=30)

    async def stats():
        try:
            return (await client.get("/stats")).json()
        except httpx.HTTPError:
            return None

    async def states():
        s = await stats()
        return [r["state"] for r in s["replicas"]] if s else None

    def chat(n=32):
        return client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": n})

    try:
        await wait(states, 10, "router up")
        assert await states() == ["off", "off"]

        resp = await chat()
        assert resp.status_code == 200, "first request was held through the cold start"

        burst = [asyncio.create_task(chat(300)) for _ in range(12)]
        async def both_ready():
            return await states() == ["ready", "ready"]
        await wait(both_ready, 10, "second replica")

        procs["auto"].send_signal(signal.SIGKILL)
        procs["auto"].wait()
        procs["auto"] = spawn("manifold.autoscaler", env)

        results = await asyncio.gather(*burst)
        assert {r.status_code for r in results} == {200}
        served = {r.headers["x-manifold-backend"] for r in results}
        assert served, "burst completed"

        async def at_zero():
            return await states() == ["off", "off"]
        await wait(at_zero, 15, "scale to zero")
        assert not list((tmp_path / "run").glob("*.pid"))

        resp = await chat()
        assert resp.status_code == 200, "second cold start"
    finally:
        await client.aclose()
        for p in procs.values():
            p.terminate()
        for p in procs.values():
            p.wait(timeout=5)
        for f in (tmp_path / "run").glob("*.pid"):
            try:
                os.killpg(int(f.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
