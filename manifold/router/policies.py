"""Routing policies as filter, score, pick tables, the shape of llm-d's endpoint picker.

A policy is data: filters that narrow the replicas, weighted scorers in [0, 1] (higher is
better), and a picker. `decide` is the only code path; policies differ only by their row in
POLICIES.
"""

from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from .prefix import BlockHash, PrefixIndex
from .replicas import Replica, ReplicaState


@dataclass(frozen=True)
class RequestCtx:
    request_id: str
    session_id: str | None
    prompt_tokens: list[int]
    block_hashes: list[BlockHash]
    est_output_tokens: int

    @property
    def load_tokens(self) -> int:
        return len(self.prompt_tokens) + self.est_output_tokens


Scorer = Callable[[RequestCtx, Replica, Sequence[Replica], PrefixIndex], float]
Filter = Callable[[Sequence[Replica], float], list[Replica]]
Picker = Callable[[Sequence[tuple[Replica, float]]], Replica]


def _lower_is_better(x: float, xs: Sequence[float]) -> float:
    lo, hi = min(xs), max(xs)
    return 1.0 if hi == lo else (hi - x) / (hi - lo)


def queue(ctx: RequestCtx, r: Replica, cands: Sequence[Replica], index: PrefixIndex) -> float:
    def waiting(x: Replica) -> float:
        return x.scraped.waiting if x.scraped else 0

    return _lower_is_better(waiting(r), [waiting(c) for c in cands])


def kv_util(ctx: RequestCtx, r: Replica, cands: Sequence[Replica], index: PrefixIndex) -> float:
    # Absolute, as llm-d's KV scorer is: min-max over two replicas would turn a 1% gap into 0 vs 1.
    return 1.0 - r.scraped.kv_usage if r.scraped else 1.0


def inflight_tokens(
    ctx: RequestCtx, r: Replica, cands: Sequence[Replica], index: PrefixIndex
) -> float:
    # Share of the busiest replica's load rather than min-max, for the same reason as kv_util.
    hi = max(c.in_flight_tokens for c in cands)
    return 1.0 if hi == 0 else 1.0 - r.in_flight_tokens / hi


def prefix(ctx: RequestCtx, r: Replica, cands: Sequence[Replica], index: PrefixIndex) -> float:
    if not ctx.block_hashes:
        return 0.0
    return index.matched(r.index, ctx.block_hashes) / len(ctx.block_hashes)


def session_slot(session_id: str, n_replicas: int) -> int:
    digest = hashlib.sha256(session_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % n_replicas


def session(ctx: RequestCtx, r: Replica, cands: Sequence[Replica], index: PrefixIndex) -> float:
    if ctx.session_id is None:
        return 0.0
    return 1.0 if r.index == session_slot(ctx.session_id, index.n_replicas) else 0.0


AFFINITY_SCORERS: tuple[Scorer, ...] = (prefix, session)


def ready(cands: Sequence[Replica], spill_factor: float) -> list[Replica]:
    return [r for r in cands if r.state is ReplicaState.READY]


def not_overloaded(cands: Sequence[Replica], spill_factor: float) -> list[Replica]:
    if not cands:
        return []
    limit = spill_factor * (min(r.in_flight_requests for r in cands) + 1)
    return [r for r in cands if r.in_flight_requests <= limit]


def argmax(scored: Sequence[tuple[Replica, float]]) -> Replica:
    # Rounding keeps float summation noise from deciding a tie.
    return min(scored, key=lambda rs: (-round(rs[1], 9), rs[0].in_flight_tokens, rs[0].index))[0]


@dataclass
class RoundRobinPicker:
    last_index: int = -1

    def __call__(self, scored: Sequence[tuple[Replica, float]]) -> Replica:
        ordered = sorted((r for r, _ in scored), key=lambda r: r.index)
        pick = next((r for r in ordered if r.index > self.last_index), ordered[0])
        self.last_index = pick.index
        return pick


@dataclass(frozen=True)
class Policy:
    name: str
    filters: tuple[Filter, ...]
    scorers: tuple[tuple[Scorer, float], ...]
    picker: Picker
    spill_factor: float = 2.0


POLICIES: dict[str, Policy] = {
    p.name: p
    for p in [
        Policy("round_robin", (ready,), (), RoundRobinPicker()),
        Policy(
            "least_load",
            (ready,),
            ((queue, 2.0), (kv_util, 2.0), (inflight_tokens, 2.0)),
            argmax,
        ),
        Policy(
            "session_hash",
            (ready, not_overloaded),
            ((session, 1.0), (inflight_tokens, 0.01)),
            argmax,
        ),
        Policy(
            "prefix_aware",
            (ready,),
            ((prefix, 3.0), (queue, 2.0), (kv_util, 2.0), (inflight_tokens, 2.0)),
            argmax,
        ),
    ]
}


def make_policy(name: str, spill_factor: float = 2.0) -> Policy:
    """A private copy, so pickers with state (round robin) are not shared between routers."""
    try:
        template = POLICIES[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; options: {sorted(POLICIES)}") from None
    return replace(template, picker=copy.deepcopy(template.picker), spill_factor=spill_factor)


class NoReplicaAvailable(Exception):
    pass


@dataclass(frozen=True)
class Decision:
    request_id: str
    session_id: str | None
    policy: str
    chosen: Replica
    scores: dict[str, dict[str, float]]
    matched_blocks: dict[str, int]
    spilled: bool
    ts: float = field(default_factory=time.time)


def decide(
    policy: Policy, ctx: RequestCtx, replicas: Sequence[Replica], index: PrefixIndex
) -> Decision:
    cands = list(replicas)
    for f in policy.filters:
        cands = f(cands, policy.spill_factor)
    if not cands:
        raise NoReplicaAvailable("no replica passed the filters")

    scores: dict[str, dict[str, float]] = {}
    for r in cands:
        row = {s.__name__: s(ctx, r, cands, index) for s, _ in policy.scorers}
        row["total"] = sum(w * row[s.__name__] for s, w in policy.scorers)
        scores[r.url] = row
    chosen = policy.picker([(r, scores[r.url]["total"]) for r in cands])

    # Affinity is judged over every ready replica, so a preferred replica that a filter
    # dropped as overloaded still counts as a spill.
    ready_replicas = ready(replicas, policy.spill_factor)
    spilled = False
    for s, _ in policy.scorers:
        if s in AFFINITY_SCORERS:
            values = {r.index: s(ctx, r, ready_replicas, index) for r in ready_replicas}
            top = max(values.values(), default=0.0)
            if top > 0 and values.get(chosen.index, 0.0) < top:
                spilled = True

    return Decision(
        request_id=ctx.request_id,
        session_id=ctx.session_id,
        policy=policy.name,
        chosen=chosen,
        scores=scores,
        matched_blocks={r.url: index.matched(r.index, ctx.block_hashes) for r in replicas},
        spilled=spilled,
    )
