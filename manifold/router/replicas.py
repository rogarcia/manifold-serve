"""Replica registry: lifecycle state, router-local in-flight load, and scraped engine gauges."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import httpx
from prometheus_client.parser import text_string_to_metric_families

log = logging.getLogger(__name__)


class ReplicaState(str, Enum):
    READY = "ready"
    DRAINING = "draining"
    SLEEPING = "sleeping"


@dataclass(frozen=True)
class EngineGauges:
    waiting: int
    running: int
    kv_usage: float
    prefix_hits_total: int
    prefix_queries_total: int
    ts: float


@dataclass
class Replica:
    url: str
    index: int
    state: ReplicaState = ReplicaState.READY
    in_flight_tokens: int = 0
    in_flight_requests: int = 0
    scraped: EngineGauges | None = None

    @contextmanager
    def in_flight(self, tokens: int) -> Iterator[None]:
        self.in_flight_tokens += tokens
        self.in_flight_requests += 1
        try:
            yield
        finally:
            self.in_flight_tokens -= tokens
            self.in_flight_requests -= 1


_SUMMED = {
    "vllm:num_requests_waiting": "waiting",
    "vllm:num_requests_running": "running",
    "vllm:prefix_cache_hits_total": "prefix_hits_total",
    "vllm:prefix_cache_queries_total": "prefix_queries_total",
}
_KV_USAGE = "vllm:kv_cache_usage_perc"


def parse_vllm_metrics(text: str, ts: float | None = None) -> EngineGauges:
    """Counts are summed across label sets; KV usage is averaged, since it is a fraction."""
    sums = dict.fromkeys(_SUMMED.values(), 0.0)
    kv: list[float] = []
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name in _SUMMED:
                sums[_SUMMED[sample.name]] += sample.value
            elif sample.name == _KV_USAGE:
                kv.append(sample.value)
    return EngineGauges(
        waiting=int(sums["waiting"]),
        running=int(sums["running"]),
        kv_usage=sum(kv) / len(kv) if kv else 0.0,
        prefix_hits_total=int(sums["prefix_hits_total"]),
        prefix_queries_total=int(sums["prefix_queries_total"]),
        ts=time.time() if ts is None else ts,
    )


async def scrape_forever(client: httpx.AsyncClient, replica: Replica, period_s: float) -> None:
    """Keeps the last good gauges on failure; the scrape-age metric shows the staleness."""
    while True:
        try:
            resp = await client.get(f"{replica.url}/metrics", timeout=max(period_s * 10, 1.0))
            resp.raise_for_status()
            replica.scraped = parse_vllm_metrics(resp.text)
        except (httpx.HTTPError, ValueError) as e:
            log.debug("scrape %s failed: %r", replica.url, e)
        await asyncio.sleep(period_s)
