"""The seam between the policy and whatever runs engines. Both methods are idempotent."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path
from typing import Protocol

import httpx

from .policy import Phase


class Fleet(Protocol):
    async def observe(self) -> tuple[Phase, ...]:
        """One phase per slot, in slot order."""

    async def scale(self, n: int) -> None:
        """Slots below n exist afterwards; slots at or above n go away."""


class ProcessFleet:
    """Engines as local processes, for Modal and the laptop.

    It keeps no state of its own: pid, ready and stop files in `run_dir` are the record, so a
    restarted autoscaler adopts running engines. A stop file also marks a stop this class did not
    start, such as an injected spot reclaim (`bench/m2/reclaim.sh`). A ready file makes READY
    sticky until the process exits: M1 saw a /metrics call take over 10 s on a healthy engine, and
    one slow probe must not take a serving replica out of the router.
    """

    def __init__(self, cmd: str, run_dir: str, urls: list[str], stop_grace_s: float = 30.0,
                 client: httpx.AsyncClient | None = None):
        self.cmd, self.run_dir, self.urls = cmd, Path(run_dir), urls
        self.stop_grace_s = stop_grace_s
        self.client = client or httpx.AsyncClient(timeout=1.0)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _pid_file(self, slot: int) -> Path:
        return self.run_dir / f"slot-{slot}.pid"

    def _stop_file(self, slot: int) -> Path:
        return self.run_dir / f"slot-{slot}.stop"

    def _ready_file(self, slot: int) -> Path:
        return self.run_dir / f"slot-{slot}.ready"

    def _pid(self, slot: int) -> int | None:
        """The live process group of a slot. Clears the records of one that has exited."""
        path = self._pid_file(slot)
        if not path.exists():
            return None
        pid = int(path.read_text())
        try:
            os.waitpid(pid, os.WNOHANG)  # an exited child stays a zombie until reaped
        except ChildProcessError:
            pass  # adopted after a restart, so it is not our child
        try:
            os.killpg(pid, 0)
            return pid
        except (ProcessLookupError, PermissionError):  # macOS: EPERM for a group of zombies
            path.unlink()
            self._stop_file(slot).unlink(missing_ok=True)
            self._ready_file(slot).unlink(missing_ok=True)
            return None

    async def _phase(self, slot: int) -> Phase:
        if self._pid(slot) is None:
            return Phase.OFF
        if self._stop_file(slot).exists():
            return Phase.STOPPING
        if self._ready_file(slot).exists():
            return Phase.READY
        try:
            resp = await self.client.get(self.urls[slot] + "/health")
        except httpx.HTTPError:
            return Phase.STARTING
        if resp.status_code != 200:
            return Phase.STARTING
        self._ready_file(slot).touch()
        return Phase.READY

    async def observe(self) -> tuple[Phase, ...]:
        return tuple(await asyncio.gather(*map(self._phase, range(len(self.urls)))))

    def _start(self, slot: int) -> None:
        self._stop_file(slot).unlink(missing_ok=True)
        log = open(self.run_dir / f"slot-{slot}.log", "ab")  # noqa: SIM115
        proc = subprocess.Popen(shlex.split(self.cmd.format(slot=slot)), stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        self._pid_file(slot).write_text(str(proc.pid))

    def _stop(self, slot: int, pid: int) -> None:
        stop = self._stop_file(slot)
        if not stop.exists():
            stop.touch()
            try:
                os.kill(pid, signal.SIGTERM)  # the leader only, as Kubernetes signals PID 1; vLLM
                # drains from there, and a group signal would reach its engine process first
            except ProcessLookupError:  # the leader died and left the rest of its group
                os.killpg(pid, signal.SIGKILL)
        elif time.time() - stop.stat().st_mtime > self.stop_grace_s:
            os.killpg(pid, signal.SIGKILL)

    async def scale(self, n: int) -> None:
        for slot in range(len(self.urls)):
            pid = self._pid(slot)
            if slot >= n and pid is not None:
                self._stop(slot, pid)
            elif slot < n and pid is None:
                self._start(slot)
            elif pid is not None and self._stop_file(slot).exists():
                self._stop(slot, pid)
