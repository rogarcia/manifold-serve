"""Admission: which request goes to which replica, and when.

vLLM never queues here (`num_requests_waiting` stayed 0 at every load in M1), so the router owns
the only queue. Replicas are picked when a request leaves the queue, not when it arrives, so a
held request is scored against the replicas that exist at that moment. One event loop writes
all of this state; nothing here needs a lock.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from .policies import Decision, RequestCtx
from .replicas import Replica, ReplicaState

MAX_PREEMPTIONS = 3


class RequestClass(str, Enum):
    INTERACTIVE = "interactive"
    BATCH = "batch"


@dataclass(frozen=True)
class Caps:
    """Per-replica in-flight limits and hold times. Unset means M1 behaviour: no limit, no hold."""

    interactive: float = math.inf
    batch: float = math.inf
    hold_interactive_s: float = 0.0
    hold_batch_s: float = 0.0

    def cap(self, cls: RequestClass) -> float:
        return self.interactive if cls is RequestClass.INTERACTIVE else self.batch

    def hold(self, cls: RequestClass) -> float:
        return self.hold_interactive_s if cls is RequestClass.INTERACTIVE else self.hold_batch_s


class Shed(Exception):
    def __init__(self, cls: RequestClass, retry_after_s: int):
        super().__init__(f"no replica available for {cls.value} within the hold limit")
        self.status = 503 if cls is RequestClass.INTERACTIVE else 429
        self.retry_after_s = retry_after_s


@dataclass(eq=False)
class Ticket:
    """One request's place in the gate. It survives replays; `decision` is set while granted."""

    ctx: RequestCtx
    cls: RequestClass
    decision: Decision | None = None
    started: bool = False
    preempted: bool = False
    preemptions: int = 0
    task: asyncio.Task | None = None
    waiter: asyncio.Future | None = None


class Gate:
    def __init__(self, replicas: Sequence[Replica], caps: Caps,
                 decide: Callable[[RequestCtx, list[Replica]], Decision]):
        self.replicas, self.caps, self._decide = replicas, caps, decide
        self._held = {c: deque[Ticket]() for c in RequestClass}
        self._running = {c: dict[Ticket, None]() for c in RequestClass}
        self.preemptions = 0

    async def acquire(self, t: Ticket, front: bool = False) -> Decision:
        """Waits for a replica. Raises Shed past the hold limit. A replay passes front=True."""
        t.waiter = asyncio.get_running_loop().create_future()
        t.preempted = False
        (self._held[t.cls].appendleft if front else self._held[t.cls].append)(t)
        self.pump()
        try:
            await asyncio.wait_for(asyncio.shield(t.waiter), self.caps.hold(t.cls))
        except asyncio.TimeoutError:
            if not t.waiter.done():  # the grant can land in the same loop pass as the timeout
                self._held[t.cls].remove(t)
                raise Shed(t.cls, math.ceil(self.caps.hold(t.cls)) or 1) from None
        except asyncio.CancelledError:
            if not t.waiter.done():
                self._held[t.cls].remove(t)
            raise
        return t.decision

    def release(self, t: Ticket) -> None:
        """Frees the slot and lets the next request in. Safe on a ticket with no grant."""
        if t.decision is None:
            return
        t.decision.chosen.give_back(t.ctx.load_tokens)
        del self._running[t.cls][t]
        t.decision = None
        self.pump()

    def pump(self) -> None:
        for cls in RequestClass:
            while self._held[cls]:
                cands = self._usable(cls)
                if not cands:
                    break
                self._grant(self._held[cls].popleft(), cands)
        self._preempt()

    def evict(self, replica: Replica) -> None:
        """A slot left service: its batch work goes back to the queue, interactive may finish."""
        for t in list(self._running[RequestClass.BATCH]):
            if t.decision.chosen is replica:
                self._cancel(t)

    def demand(self) -> dict[RequestClass, tuple[int, int]]:
        """(held, in flight) per class."""
        return {c: (len(self._held[c]), len(self._running[c])) for c in RequestClass}

    def _usable(self, cls: RequestClass) -> list[Replica]:
        now = time.monotonic()
        limit = self.caps.cap(cls)
        return [r for r in self.replicas
                if r.state is ReplicaState.READY and r.down_until <= now
                and r.in_flight_requests < limit]

    def _grant(self, t: Ticket, cands: list[Replica]) -> None:
        t.decision = self._decide(t.ctx, cands)
        t.decision.chosen.take(t.ctx.load_tokens)
        self._running[t.cls][t] = None
        t.waiter.set_result(None)

    def _preempt(self) -> None:
        held = len(self._held[RequestClass.INTERACTIVE])
        if not held:
            return
        victims = [t for t in reversed(self._running[RequestClass.BATCH])
                   if not t.started and t.preemptions < MAX_PREEMPTIONS]
        cancelling = sum(t.preempted for t in self._running[RequestClass.BATCH])
        for t in victims[:max(0, held - cancelling)]:
            if not t.preempted:
                self._cancel(t)

    def _cancel(self, t: Ticket) -> None:
        if t.started or t.preempted or t.task is None:
            return
        t.preempted = True
        t.preemptions += 1
        self.preemptions += 1
        t.task.cancel()
