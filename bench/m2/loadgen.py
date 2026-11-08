"""Open-loop load from a named trace in trace.json. Writes one JSON line per request.

Arrivals, session turns and batch prompts are all fixed by the seed before the first request, over
the whole trace, and a window only clips that schedule. So two cells with one seed send identical
requests wherever their windows overlap, and a slow server can never delay a later arrival.

--time-scale K divides trace time by K and batch sizes by K. Rates stay the same, because the
engine's token time is not scaled, so the load an engine sees stays the same too. t_* fields are
Unix seconds; trace_t is the arrival in trace seconds. <out>.meta.json maps one to the other.
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import json
import math
import multiprocessing
import os
import random
import re
import resource
import statistics
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

TRACE_FILE = Path(__file__).with_name("trace.json")
CHARS_PER_TOKEN = 4
# Stands in for response time when picking an idle session, so the pick ignores the server.
VIRTUAL_S_PER_TOKEN = 0.03
MAX_CONNECTIONS = 4096
TAIL_BYTES = 1024
TRUNCATED = b'"type": "truncated"'
USAGE = re.compile(rb'"completion_tokens":\s*(\d+)')


@dataclass(frozen=True)
class Arrival:
    t: float
    cls: str
    session: str | None
    messages: list[dict]
    max_tokens: int


def rate_at(phases: list[dict], t: float) -> float:
    for p in phases:
        if p["t0"] <= t < p["t1"]:
            return p["rps0"] + (p["rps1"] - p["rps0"]) * (t - p["t0"]) / (p["t1"] - p["t0"])
    return 0.0


def arrival_times(phases: list[dict], rng: random.Random, scale: float) -> list[float]:
    """Poisson by thinning, in trace seconds."""
    out = []
    for p in phases:
        peak = max(p["rps0"], p["rps1"]) / scale
        t = p["t0"]
        while peak > 0:
            t += rng.expovariate(peak)
            if t >= p["t1"]:
                break
            if rng.random() * peak <= rate_at([p], t) / scale:
                out.append(t)
    return out


def answer_tokens(answer: dict) -> int:
    return max(1, len(answer["content"]) // CHARS_PER_TOKEN)


def turns(conv: dict) -> Iterator[tuple[list[dict], int]]:
    messages = conv["messages"]
    for i in range(0, len(messages) - 1, 2):
        yield messages[:i + 1], answer_tokens(messages[i + 1])


def build_schedule(trace: dict, conversations: list[dict], seed: int, sessions: int,
                   scale: float) -> list[Arrival]:
    rng = random.Random(seed)
    order = list(conversations)
    rng.shuffle(order)
    n_batch = sum(math.ceil(b["n"] / scale) for b in trace["batch"])
    pool, batch_pool = iter(order[:len(order) - n_batch]), iter(order[len(order) - n_batch:])

    def fresh() -> tuple[str, Iterator]:
        conv = next(pool)
        return conv["id"], turns(conv)

    live = [fresh() for _ in range(sessions)]
    idle = [(0.0, i) for i in range(sessions)]
    out = []
    for t in arrival_times(trace["phases"], rng, scale):
        free_at, i = heapq.heappop(idle)
        turn = next(live[i][1], None)
        if turn is None:
            live[i] = fresh()
            turn = next(live[i][1])
        out.append(Arrival(t, "interactive", live[i][0], *turn))
        busy = turn[1] * VIRTUAL_S_PER_TOKEN * scale
        heapq.heappush(idle, (max(t, free_at) + busy, i))
    for b in trace["batch"]:
        for _ in range(math.ceil(b["n"] / scale)):
            messages = next(batch_pool)["messages"]
            out.append(Arrival(b["t"], "batch", None, messages[:1], answer_tokens(messages[1])))
    return sorted(out, key=lambda a: a.t)


def clip(schedule: list[Arrival], start: float, end: float) -> list[Arrival]:
    return [a for a in schedule if start <= a.t < end]


def request_body(a: Arrival, model: str) -> dict:
    body = {"model": model, "messages": a.messages, "max_tokens": a.max_tokens}
    if a.cls == "batch":
        return {**body, "stream": False}
    return {**body, "stream": True, "stream_options": {"include_usage": True}}


def request_headers(a: Arrival) -> dict:
    return {"x-manifold-class": "batch"} if a.cls == "batch" else {"x-session-id": a.session}


async def send(client: httpx.AsyncClient, a: Arrival, model: str, rec: dict) -> None:
    tail, t_last = b"", None
    rec["t_sent"] = time.time()
    async with client.stream("POST", "/v1/chat/completions", json=request_body(a, model),
                             headers=request_headers(a)) as resp:
        rec["status"] = resp.status_code
        async for raw in resp.aiter_raw():
            t_last = time.time()
            rec["t_first"] = rec["t_first"] or t_last
            rec["n_chunks"] += raw.count(b"data: ")
            tail = (tail + raw)[-TAIL_BYTES:]
            rec["truncated"] = rec["truncated"] or TRUNCATED in tail
    if a.cls == "batch" or resp.status_code != 200:
        return
    rec["truncated"] = rec["truncated"] or b"[DONE]" not in tail
    usage = USAGE.search(tail)
    rec["completion_tokens"] = int(usage.group(1)) if usage else None
    if usage and int(usage.group(1)) > 1:
        rec["tpot_ms"] = (t_last - rec["t_first"]) * 1000 / (int(usage.group(1)) - 1)


async def one(client: httpx.AsyncClient, a: Arrival, t_scheduled: float, model: str,
              timeout_s: float, fd: int) -> None:
    rec = {"class": a.cls, "session": a.session, "trace_t": a.t, "t_scheduled": t_scheduled,
           "t_sent": None, "t_first": None, "t_done": None, "status": None, "n_chunks": 0,
           "completion_tokens": None, "tpot_ms": None, "truncated": False, "error": None}
    try:
        await asyncio.wait_for(send(client, a, model, rec), timeout_s)
    except (httpx.HTTPError, TimeoutError) as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["truncated"] = rec["status"] == 200 and rec["t_first"] is not None
    rec["t_done"] = time.time()
    os.write(fd, (json.dumps(rec) + "\n").encode())


async def dispatch(arrivals: list[Arrival], when, args: argparse.Namespace, fd: int) -> None:
    timeouts = {"interactive": args.timeout_interactive, "batch": args.timeout_batch}
    # Held requests stay connected, so the limit has to exceed every request held at once.
    limits = httpx.Limits(max_connections=MAX_CONNECTIONS // args.workers,
                          max_keepalive_connections=256)
    async with httpx.AsyncClient(base_url=args.url, limits=limits,
                                 timeout=httpx.Timeout(None, connect=30.0)) as client:
        tasks = []
        for a in arrivals:
            t_scheduled = when(a)
            await asyncio.sleep(max(0.0, t_scheduled - time.time()))
            tasks.append(asyncio.create_task(
                one(client, a, t_scheduled, args.model, timeouts[a.cls], fd)))
        await asyncio.gather(*tasks)


def worker(arrivals: list[Arrival], t0: float, start: float, args: argparse.Namespace,
           fd: int) -> None:
    asyncio.run(dispatch(arrivals, lambda a: t0 + (a.t - start) / args.time_scale, args, fd))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--trace", default="main")
    p.add_argument("--window", help="a name under the trace's windows; default the whole trace")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sessions", type=int, default=384)
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=8,
                   help="processes; streaming hundreds of responses saturates one event loop")
    p.add_argument("--timeout-interactive", type=float, default=640.0)
    p.add_argument("--timeout-batch", type=float, default=1500.0)
    args = p.parse_args()

    trace = json.loads(TRACE_FILE.read_text())[args.trace]
    full = (0, trace["phases"][-1]["t1"])
    start, end = trace["windows"][args.window] if args.window else full
    conversations = json.loads(Path(args.data).read_text())
    schedule = clip(build_schedule(trace, conversations, args.seed, args.sessions,
                                   args.time_scale), start, end)

    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = MAX_CONNECTIONS + 64
    resource.setrlimit(resource.RLIMIT_NOFILE,
                       (want if hard == resource.RLIM_INFINITY else min(want, hard), hard))
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    t0 = time.time() + 2.0
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=worker, args=(schedule[i::args.workers], t0, start, args, fd))
             for i in range(args.workers)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    recs = [json.loads(line) for line in Path(args.out).read_text().splitlines()]
    lag_ms = sorted((r["t_sent"] - r["t_scheduled"]) * 1000 for r in recs if r["t_sent"])
    meta = {"trace": args.trace, "window": args.window, "start": start, "end": end,
            "seed": args.seed, "time_scale": args.time_scale, "t0": t0, "workers": args.workers,
            "n_scheduled": len(schedule), "n_recorded": len(recs),
            "lag_ms": {"p50": statistics.median(lag_ms) if lag_ms else None,
                       "p99": lag_ms[int(len(lag_ms) * 0.99)] if lag_ms else None,
                       "max": lag_ms[-1] if lag_ms else None}}
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta))
    if len(recs) != len(schedule) or any(proc.exitcode for proc in procs):
        raise SystemExit(f"loadgen: {len(recs)} records of {len(schedule)} scheduled")


if __name__ == "__main__":
    main()
