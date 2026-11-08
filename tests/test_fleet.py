"""ProcessFleet against real child processes, and the loop against a real router and sims."""

import asyncio
import os
import sys

import httpx

from manifold.autoscaler.fleet import ProcessFleet
from manifold.autoscaler.policy import Phase

SLEEPER = f"{sys.executable} -c 'import time; time.sleep(60)'"
STUBBORN = (f"{sys.executable} -c 'import signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'")


class NoHealth(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        raise httpx.ConnectError("refused", request=request)


class Healthy(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        return httpx.Response(200)


def fleet(tmp_path, cmd=SLEEPER, transport=None, **kw):
    client = httpx.AsyncClient(transport=transport or NoHealth())
    return ProcessFleet(cmd, str(tmp_path), ["http://a", "http://b"], client=client, **kw)


async def eventually(f, phases, timeout=3.0):
    for _ in range(int(timeout / 0.02)):
        if await f.observe() == phases:
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"phases {await f.observe()} != {phases}")


async def test_scale_is_idempotent_and_reports_phases(tmp_path):
    f = fleet(tmp_path)
    await f.scale(1)
    pid = (tmp_path / "slot-0.pid").read_text()
    await f.scale(1)
    assert (tmp_path / "slot-0.pid").read_text() == pid
    assert await f.observe() == (Phase.STARTING, Phase.OFF)
    await f.scale(0)
    await eventually(f, (Phase.OFF, Phase.OFF))


async def test_a_healthy_engine_is_ready(tmp_path):
    f = fleet(tmp_path, transport=Healthy())
    await f.scale(1)
    assert await f.observe() == (Phase.READY, Phase.OFF)
    await f.scale(0)


async def test_one_failed_probe_does_not_take_a_ready_engine_out(tmp_path):
    f = fleet(tmp_path, transport=Healthy())
    await f.scale(1)
    assert await f.observe() == (Phase.READY, Phase.OFF)
    f.client = httpx.AsyncClient(transport=NoHealth())
    assert await f.observe() == (Phase.READY, Phase.OFF)
    await f.scale(0)


async def test_a_stale_stop_file_does_not_mark_a_new_engine_as_stopping(tmp_path):
    (tmp_path / "slot-0.stop").touch()
    f = fleet(tmp_path, transport=Healthy())
    await f.scale(1)
    assert await f.observe() == (Phase.READY, Phase.OFF)
    await f.scale(0)


async def test_a_restarted_autoscaler_adopts_running_engines(tmp_path):
    first = fleet(tmp_path, transport=Healthy())
    await first.scale(2)
    second = fleet(tmp_path, transport=Healthy())
    assert await second.observe() == (Phase.READY, Phase.READY)
    pids = [(tmp_path / f"slot-{i}.pid").read_text() for i in range(2)]
    await second.scale(2)
    assert [(tmp_path / f"slot-{i}.pid").read_text() for i in range(2)] == pids
    await second.scale(0)


async def test_an_external_stop_shows_as_stopping_then_restarts(tmp_path):
    f = fleet(tmp_path, transport=Healthy())
    await f.scale(1)
    (tmp_path / "slot-0.stop").touch()
    assert await f.observe() == (Phase.STOPPING, Phase.OFF)
    os.killpg(int((tmp_path / "slot-0.pid").read_text()), 9)
    await eventually(f, (Phase.OFF, Phase.OFF))
    await f.scale(1)
    assert await f.observe() == (Phase.READY, Phase.OFF)
    await f.scale(0)


async def test_an_engine_that_ignores_sigterm_is_killed_after_the_grace(tmp_path):
    f = fleet(tmp_path, cmd=STUBBORN, stop_grace_s=0.0)
    await f.scale(1)
    await asyncio.sleep(0.5)
    await f.scale(0)
    await asyncio.sleep(0.05)
    await f.scale(0)
    await eventually(f, (Phase.OFF, Phase.OFF))


async def test_a_group_that_outlived_its_leader_is_stopped(tmp_path):
    f = fleet(tmp_path, cmd="sh -c 'sleep 60 & exit 0'")
    await f.scale(1)
    pid = int((tmp_path / "slot-0.pid").read_text())
    while True:
        await asyncio.sleep(0.05)
        f._pid(0)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    await f.scale(0)
    for _ in range(100):
        if (await f.observe())[0] is Phase.OFF:
            break
        await asyncio.sleep(0.05)
    assert (await f.observe())[0] is Phase.OFF
