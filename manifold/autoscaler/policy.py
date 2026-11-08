"""The scaling decision: observations in, two integers out. No I/O and no clock.

The whole fleet decision is `Target(serve, run)`. Slots are a stack: slots below `serve` take
work, slots in [serve, run) drain, and the rest are off. The top slot is always the one removed,
which is also what a Kubernetes StatefulSet does when it scales down.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

from manifold.router.replicas import ReplicaState


class Phase(str, Enum):
    OFF = "off"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"


@dataclass(frozen=True)
class Signal:
    """Held plus in-flight requests per class, as the router counts them."""

    interactive: int
    batch: int
    in_flight: tuple[int, ...]

    @property
    def demand(self) -> int:
        return self.interactive + self.batch


@dataclass(frozen=True)
class Observation:
    now: float
    phases: tuple[Phase, ...]
    signal: Signal


@dataclass(frozen=True)
class Spec:
    max_replicas: int = 2
    floor: int = 0
    tick_s: float = 1.0
    target_interactive: float = 26.0
    batch_rps_per_replica: float = 9.0
    batch_deadline_s: float = 600.0
    up_window_s: float = 10.0
    settle_s: float = 60.0
    down_window_s: float = 120.0
    down_ratio: float = 0.7
    idle_to_zero_s: float = 170.0
    drain_deadline_s: float = 30.0


@dataclass(frozen=True)
class Target:
    serve: int
    run: int

    def __post_init__(self) -> None:
        assert 0 <= self.serve <= self.run


@dataclass(frozen=True)
class Memo:
    """Timers only. Losing them delays a change and never causes one."""

    serve: int
    n_ready: int = 0
    ready_at: float = -math.inf
    above_since: float | None = None
    below_since: float | None = None
    idle_since: float | None = None
    drain_since: float | None = None

    @classmethod
    def recover(cls, obs: Observation) -> Memo:
        """After a restart, whatever exists keeps serving until the timers run again."""
        alive = sum(p is not Phase.OFF for p in obs.phases)
        return cls(serve=alive, n_ready=sum(p is Phase.READY for p in obs.phases))


def replicas_needed(s: Signal, spec: Spec) -> float:
    return s.interactive / spec.target_interactive \
        + s.batch / spec.batch_deadline_s / spec.batch_rps_per_replica


def _since(start: float | None, now: float, active: bool) -> float | None:
    return (now if start is None else start) if active else None


def _held_for(start: float | None, now: float, window: float) -> bool:
    return start is not None and now - start >= window


def _next_serve(obs: Observation, spec: Spec, m: Memo) -> Memo:
    now, load = obs.now, replicas_needed(obs.signal, spec)
    idle = obs.signal.demand == 0
    wanted = min(math.ceil(load), spec.max_replicas)
    m = replace(
        m,
        above_since=_since(m.above_since, now, wanted > m.serve),
        below_since=_since(m.below_since, now, load <= spec.down_ratio * (m.serve - 1)),
        idle_since=_since(m.idle_since, now, idle),
    )

    if m.serve == 0 and not idle:
        return replace(m, serve=1, above_since=None)
    starting = Phase.STARTING in obs.phases
    if (_held_for(m.above_since, now, spec.up_window_s) and not starting
            and now - m.ready_at >= spec.settle_s):
        return replace(m, serve=m.serve + 1, above_since=None)
    if m.serve >= 1 and _held_for(m.idle_since, now, spec.idle_to_zero_s):
        return replace(m, serve=0, below_since=None, idle_since=None)
    if m.serve >= 2 and not starting and _held_for(m.below_since, now, spec.down_window_s):
        return replace(m, serve=m.serve - 1, below_since=None)
    return m


def _next_run(obs: Observation, spec: Spec, m: Memo) -> tuple[int, Memo]:
    """Keep a draining slot until it is empty or its deadline passes."""
    alive = max((i + 1 for i, p in enumerate(obs.phases) if p is not Phase.OFF), default=0)
    run = max(alive, m.serve)
    drain_since = _since(m.drain_since, obs.now, run > m.serve)
    if _held_for(drain_since, obs.now, spec.drain_deadline_s):
        return m.serve, replace(m, drain_since=None)
    while run > m.serve and obs.signal.in_flight[run - 1] == 0:
        run -= 1
    return run, replace(m, drain_since=drain_since if run > m.serve else None)


def step(obs: Observation, spec: Spec, memo: Memo) -> tuple[Target, Memo]:
    n_ready = sum(p is Phase.READY for p in obs.phases)
    ready_at = obs.now if n_ready > memo.n_ready else memo.ready_at
    m = _next_serve(obs, spec, replace(memo, n_ready=n_ready, ready_at=ready_at))
    m = replace(m, serve=max(m.serve, spec.floor))
    run, m = _next_run(obs, spec, m)
    return Target(m.serve, run), m


def slot_states(target: Target, phases: Sequence[Phase]) -> list[ReplicaState]:
    def state(i: int, p: Phase) -> ReplicaState:
        if p is Phase.READY and i < target.serve:
            return ReplicaState.READY
        return ReplicaState.DRAINING if p in (Phase.READY, Phase.STOPPING) else ReplicaState.OFF

    return [state(i, p) for i, p in enumerate(phases)]
