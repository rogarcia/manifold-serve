"""The control loop. Read the fleet and the router, ask the policy, apply the answer.

    MANIFOLD_ROUTER_URL  router base URL (http://127.0.0.1:8000)
    MANIFOLD_SPEC        JSON object of `Spec` fields ({})
    MANIFOLD_FLEET       process | kube (process)
    MANIFOLD_TICK_LOG    JSONL path, one line per tick (none)
    process fleet: MANIFOLD_ENGINE_CMD, with {slot}; MANIFOLD_RUN_DIR
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import IO

import httpx

from .fleet import Fleet, ProcessFleet
from .policy import Memo, Observation, Signal, Spec, slot_states, step

log = logging.getLogger("manifold.autoscaler")


class RouterClient:
    def __init__(self, url: str, client: httpx.AsyncClient):
        self.url, self.client = url, client

    async def urls(self) -> list[str]:
        resp = await self.client.get(f"{self.url}/stats")
        resp.raise_for_status()
        return [r["url"] for r in resp.json()["replicas"]]

    async def signal(self) -> Signal | None:
        """None when the router cannot answer: skip the tick and change nothing."""
        try:
            resp = await self.client.get(f"{self.url}/stats")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("router unreachable: %r", e)
            return None
        stats = resp.json()
        d = stats["demand"]
        return Signal(
            interactive=d["interactive"]["held"] + d["interactive"]["in_flight"],
            batch=d["batch"]["held"] + d["batch"]["in_flight"],
            in_flight=tuple(r["in_flight_requests"] for r in stats["replicas"]),
        )

    async def push(self, states: list) -> None:
        for i, state in enumerate(states):
            await self.client.post(f"{self.url}/admin/replicas/{i}/state",
                                   json={"state": state.value})


async def run(fleet: Fleet, router: RouterClient, spec: Spec, tick_log: IO[str] | None) -> None:
    memo: Memo | None = None
    while True:
        try:
            phases = await fleet.observe()
            signal = await router.signal()
            if signal is not None:
                obs = Observation(time.time(), phases, signal)
                memo = memo or Memo.recover(obs)
                target, memo = step(obs, spec, memo)
                await fleet.scale(target.run)
                await router.push(slot_states(target, phases))
                if tick_log:
                    tick_log.write(json.dumps({
                        "t": obs.now, "phases": [p.value for p in phases],
                        "signal": asdict(signal), "serve": target.serve, "run": target.run,
                    }) + "\n")
                    tick_log.flush()
        except httpx.HTTPError as e:
            log.warning("tick abandoned: %r", e)
        await asyncio.sleep(spec.tick_s)


def _make_fleet(env: dict[str, str], urls: list[str]) -> Fleet:
    kind = env.get("MANIFOLD_FLEET", "process")
    if kind == "process":
        return ProcessFleet(env["MANIFOLD_ENGINE_CMD"], env["MANIFOLD_RUN_DIR"], urls)
    if kind == "kube":
        from .kube import KubeFleet

        return KubeFleet.in_cluster(len(urls))
    raise SystemExit(f"manifold-autoscaler: unknown MANIFOLD_FLEET {kind!r}")


def _open_tick_log(path: str | None) -> IO[str] | None:
    return Path(path).open("a", encoding="utf-8") if path else None


async def amain(env: dict[str, str]) -> None:
    spec = Spec(**json.loads(env.get("MANIFOLD_SPEC", "{}")))
    url = env.get("MANIFOLD_ROUTER_URL", "http://127.0.0.1:8000").rstrip("/")
    async with httpx.AsyncClient(timeout=5.0) as client:
        router = RouterClient(url, client)
        while True:
            try:
                urls = await router.urls()
                break
            except httpx.HTTPError:
                log.info("waiting for the router at %s", url)
                await asyncio.sleep(1)
        if len(urls) != spec.max_replicas:
            raise SystemExit(f"manifold-autoscaler: router has {len(urls)} slots, spec says "
                             f"{spec.max_replicas}")
        fleet = _make_fleet(env, urls)
        await run(fleet, router, spec, _open_tick_log(env.get("MANIFOLD_TICK_LOG")))


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(asctime)s [autoscaler] %(message)s")
    try:
        asyncio.run(amain(dict(os.environ)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
