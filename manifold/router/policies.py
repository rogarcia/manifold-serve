"""Routing policies.

A policy picks a backend for each incoming request. Backends track the number of
outstanding (in-flight) requests, which is the cheapest useful load signal: it is
what the router can know without scraping backend metrics on the hot path.

Policies implemented:

- round_robin:        classic baseline, ignores load entirely.
- least_outstanding:  pick the backend with the fewest in-flight requests.
- prefix_aware:       consistent-hash the conversation prefix to a preferred backend
                      so multi-turn sessions keep hitting the replica that already
                      holds their KV-cache blocks; fall back to least-outstanding
                      when the preferred backend is overloaded relative to the rest.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field


@dataclass
class Backend:
    url: str
    outstanding: int = 0
    total_requests: int = field(default=0)

    def acquire(self) -> None:
        self.outstanding += 1
        self.total_requests += 1

    def release(self) -> None:
        self.outstanding = max(0, self.outstanding - 1)


class Policy:
    name = "base"

    def __init__(self, backends: list[Backend]):
        if not backends:
            raise ValueError("at least one backend required")
        self.backends = backends

    def select(self, prefix_key: str | None = None) -> Backend:
        raise NotImplementedError


class RoundRobin(Policy):
    name = "round_robin"

    def __init__(self, backends: list[Backend]):
        super().__init__(backends)
        self._cycle = itertools.cycle(self.backends)

    def select(self, prefix_key: str | None = None) -> Backend:
        return next(self._cycle)


class LeastOutstanding(Policy):
    name = "least_outstanding"

    def select(self, prefix_key: str | None = None) -> Backend:
        return min(self.backends, key=lambda b: b.outstanding)


class PrefixAware(Policy):
    """Session-affinity routing for KV-cache reuse.

    `overload_factor` controls how much extra queueing we tolerate on the preferred
    backend before spilling: affinity is worthless if it means waiting behind a deep
    queue while another replica sits idle, so the interesting question (and the M1
    benchmark) is where that break-even point sits for a given model and load.
    """

    name = "prefix_aware"

    def __init__(self, backends: list[Backend], overload_factor: float = 2.0):
        super().__init__(backends)
        self.overload_factor = overload_factor

    def select(self, prefix_key: str | None = None) -> Backend:
        fallback = min(self.backends, key=lambda b: b.outstanding)
        if not prefix_key:
            return fallback
        digest = hashlib.sha256(prefix_key.encode()).digest()
        preferred = self.backends[int.from_bytes(digest[:8], "big") % len(self.backends)]
        if preferred.outstanding <= self.overload_factor * (fallback.outstanding + 1):
            return preferred
        return fallback


POLICIES: dict[str, type[Policy]] = {
    RoundRobin.name: RoundRobin,
    LeastOutstanding.name: LeastOutstanding,
    PrefixAware.name: PrefixAware,
}


def make_policy(name: str, backend_urls: list[str]) -> Policy:
    try:
        cls = POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; options: {sorted(POLICIES)}") from None
    return cls([Backend(url=u.rstrip("/")) for u in backend_urls])
