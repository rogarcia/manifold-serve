"""Open-loop async load generator for OpenAI-compatible endpoints.

Design choices that matter for honest benchmarks:

- Open-loop (Poisson) arrivals: requests are launched on a schedule independent of
  completions, so the system under test cannot hide overload by slowing the client
  (the coordinated-omission trap that closed-loop loadgens fall into).
- Multi-turn sessions with a shared system prompt: consecutive requests in a session
  share a growing prefix, which is what real chat traffic looks like and what makes
  prefix-cache-aware routing measurable at the router tier.
- TTFT measured at first SSE chunk, per request; results dumped to CSV for analysis.

Usage:
    python -m manifold.loadgen --base-url http://localhost:8000 \
        --model Inferact/Qwen3.8-27B-NVFP4 --rate 4 --duration 120 \
        --sessions 32 --out bench/results.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass, field

import httpx

SYSTEM_PROMPT = (
    "You are a concise assistant helping benchmark an inference platform. "
    "Answer in no more than three sentences."
)

TOPICS = [
    "the history of railway signaling",
    "how continuous batching works in LLM serving",
    "why p99 latency matters more than the mean",
    "the tradeoffs of consistent hashing",
    "how KV-caches grow with context length",
    "what makes autoscaling stateful services hard",
    "canary deployments versus blue-green",
    "the physics of GPU memory bandwidth",
]


@dataclass
class Session:
    session_id: int
    topic: str
    history: list[dict] = field(default_factory=list)

    def next_messages(self, turn: int) -> list[dict]:
        user = {"role": "user", "content": f"Question {turn + 1} about {self.topic}: tell me one more thing."}
        self.history.append(user)
        return [{"role": "system", "content": SYSTEM_PROMPT}, *self.history]

    def record_reply(self, text: str) -> None:
        self.history.append({"role": "assistant", "content": text})


@dataclass
class Result:
    session_id: int
    turn: int
    start: float
    ttft_ms: float | None
    total_ms: float
    output_chars: int
    output_tokens: int
    status: int
    backend: str


async def one_request(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    session: Session,
    turn: int,
    results: list[Result],
    max_tokens: int = 120,
    disable_thinking: bool = False,
) -> None:
    messages = session.next_messages(turn)
    payload: dict = {"model": model, "messages": messages, "stream": True, "max_tokens": max_tokens}
    if disable_thinking:
        # Qwen3.x thinking models emit <think> first; for latency benchmarks measure answer tokens.
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    start = time.perf_counter()
    ttft = None
    text_parts: list[str] = []
    n_chunks = 0
    status = 0
    backend = ""
    try:
        async with client.stream("POST", f"{base_url}/v1/chat/completions", json=payload) as resp:
            status = resp.status_code
            backend = resp.headers.get("x-manifold-backend", "")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content") or ""
                    if delta and ttft is None:
                        ttft = (time.perf_counter() - start) * 1000
                    if delta:
                        n_chunks += 1
                        text_parts.append(delta)
                except (KeyError, IndexError, json.JSONDecodeError):
                    continue
    except httpx.HTTPError:
        status = -1
    total_ms = (time.perf_counter() - start) * 1000
    reply = "".join(text_parts)
    session.record_reply(reply)
    results.append(
        Result(session.session_id, turn, start, ttft, total_ms, len(reply), n_chunks, status, backend)
    )


async def run(args: argparse.Namespace) -> list[Result]:
    sessions = [Session(i, random.choice(TOPICS)) for i in range(args.sessions)]
    turns = {s.session_id: 0 for s in sessions}
    results: list[Result] = []
    tasks: set[asyncio.Task] = set()
    deadline = time.perf_counter() + args.duration

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        while time.perf_counter() < deadline:
            await asyncio.sleep(random.expovariate(args.rate))
            session = random.choice(sessions)
            turn = turns[session.session_id]
            turns[session.session_id] += 1
            task = asyncio.create_task(
                one_request(
                    client, args.base_url, args.model, session, turn, results,
                    max_tokens=args.max_tokens, disable_thinking=args.disable_thinking,
                )
            )
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    return results


def summarize(results: list[Result]) -> None:
    ok = [r for r in results if r.status == 200 and r.ttft_ms is not None]
    print(f"\nrequests: {len(results)}  ok: {len(ok)}  errors: {len(results) - len(ok)}")
    if not ok:
        return
    def pct(values: list[float], q: float) -> float:
        return values[min(len(values) - 1, int(q * len(values)))]

    for label, values in (
        ("TTFT ms", sorted(r.ttft_ms for r in ok)),
        ("total ms", sorted(r.total_ms for r in ok)),
    ):
        print(
            f"{label:9s} p50={pct(values, 0.50):8.1f}  p90={pct(values, 0.90):8.1f}  "
            f"p99={pct(values, 0.99):8.1f}  mean={statistics.fmean(values):8.1f}"
        )
    span = max(r.start + r.total_ms / 1000 for r in ok) - min(r.start for r in ok)
    tokens = sum(r.output_tokens for r in ok)
    if span > 0:
        print(f"output tok/s (stream chunks): {tokens / span:8.1f}   total output tokens: {tokens}")
    decode = [
        (r.total_ms - r.ttft_ms) / (r.output_tokens - 1)
        for r in ok
        if r.output_tokens > 1 and r.ttft_ms
    ]
    if decode:
        decode.sort()
        print(
            f"TPOT ms   p50={pct(decode, 0.50):8.1f}  p90={pct(decode, 0.90):8.1f}  "
            f"p99={pct(decode, 0.99):8.1f}"
        )
    by_backend: dict[str, int] = {}
    for r in ok:
        by_backend[r.backend] = by_backend.get(r.backend, 0) + 1
    if any(by_backend):
        print("backend distribution:", by_backend)


def write_csv(results: list[Result], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["session_id", "turn", "start", "ttft_ms", "total_ms", "output_chars",
             "output_tokens", "status", "backend"]
        )
        for r in results:
            writer.writerow(
                [r.session_id, r.turn, f"{r.start:.6f}", r.ttft_ms, f"{r.total_ms:.1f}",
                 r.output_chars, r.output_tokens, r.status, r.backend]
            )
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="Inferact/Qwen3.8-27B-NVFP4")
    parser.add_argument("--rate", type=float, default=2.0, help="mean arrivals per second (Poisson)")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds")
    parser.add_argument("--sessions", type=int, default=16, help="concurrent multi-turn sessions")
    parser.add_argument("--out", default="", help="CSV output path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument(
        "--disable-thinking", action="store_true",
        help="send chat_template_kwargs.enable_thinking=false (Qwen3.x thinking models)",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    results = asyncio.run(run(args))
    summarize(results)
    if args.out:
        write_csv(results, args.out)


if __name__ == "__main__":
    main()
