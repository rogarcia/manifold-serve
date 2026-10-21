"""A fake vLLM engine: scheduler slots, a block-level LRU prefix cache, and timed generation."""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from manifold.router.prefix import BlockHash, block_hashes

_WORDS = ("the", "cache", "router", "block", "token", "prefix", "replica", "queue", "warm",
          "cold", "session", "turn", "load", "spill", "hit", "miss")
_TIME_BUCKETS = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


@dataclass(frozen=True)
class SimConfig:
    model: str = "sim-model"
    block_tokens: int = 16
    kv_blocks: int = 4096
    max_num_seqs: int = 16
    prefill_us_per_token: float = 50.0
    decode_ms_per_token: float = 8.0
    seed: int = 0


class _Metrics:
    def __init__(self, model: str):
        self.registry = r = CollectorRegistry()
        labels = ["model_name"]

        def gauge(name: str, doc: str) -> Gauge:
            return Gauge(name, doc, labels, registry=r)

        self.waiting = gauge("vllm:num_requests_waiting", "Requests waiting to be scheduled")
        self.running = gauge("vllm:num_requests_running", "Requests running on the engine")
        self.kv_usage = gauge("vllm:kv_cache_usage_perc", "KV-cache usage, 1 means 100 percent")
        self.queries = Counter("vllm:prefix_cache_queries", "Prefix-cache queried tokens", labels,
                               registry=r).labels(model)
        self.hits = Counter("vllm:prefix_cache_hits", "Prefix-cache hit tokens", labels,
                            registry=r).labels(model)
        self.queue_time = Histogram("vllm:request_queue_time_seconds", "Time in WAITING",
                                    labels, registry=r, buckets=_TIME_BUCKETS).labels(model)
        self.prefill_time = Histogram("vllm:request_prefill_time_seconds", "Time in prefill",
                                      labels, registry=r, buckets=_TIME_BUCKETS).labels(model)


class Engine:
    def __init__(self, config: SimConfig):
        self.config = config
        self.metrics = _Metrics(config.model)
        self._cache: OrderedDict[BlockHash, None] = OrderedDict()
        self._slots = asyncio.Semaphore(config.max_num_seqs)
        self._rng = random.Random(config.seed)
        self.waiting = 0
        self.running = 0
        self._held_blocks = 0

    @property
    def kv_usage(self) -> float:
        return min(1.0, self._held_blocks / self.config.kv_blocks)

    def reset_prefix_cache(self) -> None:
        self._cache.clear()

    def _cached_prefix_blocks(self, blocks: list[BlockHash]) -> int:
        n = 0
        for h in blocks:
            if h not in self._cache:
                break
            self._cache.move_to_end(h)
            n += 1
        return n

    def _store(self, blocks: list[BlockHash]) -> None:
        for h in blocks:
            self._cache[h] = None
            self._cache.move_to_end(h)
        while len(self._cache) > self.config.kv_blocks:
            self._cache.popitem(last=False)

    async def generate(self, tokens: list[int], max_tokens: int) -> AsyncIterator[str]:
        """Yields output token texts: the first after queue wait plus uncached prefill."""
        cfg = self.config
        t_arrived = time.monotonic()
        self.waiting += 1
        try:
            await self._slots.acquire()
        finally:
            self.waiting -= 1
        self.running += 1
        held = math.ceil(len(tokens) / cfg.block_tokens)
        self._held_blocks += held
        try:
            t_scheduled = time.monotonic()
            self.metrics.queue_time.observe(t_scheduled - t_arrived)
            blocks = block_hashes(tokens, cfg.block_tokens)
            hit_tokens = self._cached_prefix_blocks(blocks) * cfg.block_tokens
            self.metrics.queries.inc(len(tokens))
            self.metrics.hits.inc(hit_tokens)
            await asyncio.sleep((len(tokens) - hit_tokens) * cfg.prefill_us_per_token / 1e6)
            self._store(blocks)
            self.metrics.prefill_time.observe(time.monotonic() - t_scheduled)

            output = bytearray()
            for i in range(max_tokens):
                if i:
                    await asyncio.sleep(cfg.decode_ms_per_token / 1000)
                text = self._rng.choice(_WORDS) + " "
                output += text.encode()
                grown = math.ceil((len(tokens) + len(output)) / cfg.block_tokens)
                self._held_blocks += grown - held
                held = grown
                yield text
            # The reply bytes are what the client renders back into the next turn's prompt,
            # so caching them gives follow-up turns the hit vLLM would give them.
            self._store(block_hashes(tokens + list(output), cfg.block_tokens))
        finally:
            self._held_blocks -= held
            self.running -= 1
            self._slots.release()

    def render_metrics(self) -> bytes:
        m, model = self.metrics, self.config.model
        m.waiting.labels(model).set(self.waiting)
        m.running.labels(model).set(self.running)
        m.kv_usage.labels(model).set(self.kv_usage)
        return generate_latest(m.registry)
