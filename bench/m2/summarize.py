"""Metrics and pass rules P1 to P8 for one M2 run.

Usage: summarize.py RUN_DIR     RUN_DIR holds one directory per cell, named c1-..., c2-... .
Only cells with DONE count. A rule whose cell is missing prints NOT EVALUATED.
Trace seconds index the phases; wall seconds index ticks, decisions and samples. Each cell's
requests.meta.json maps one to the other, so a time-compressed run is read the same way.
"""

from __future__ import annotations

import json
import sys
from bisect import bisect_right
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from loadgen import TRACE_FILE, rate_at

PASS, FAIL, NOT_EVALUATED = "PASS", "FAIL", "NOT EVALUATED"
TTFT_LIMIT_MS, TPOT_LIMIT_MS = 500.0, 40.0
TRACE_S = 1080.0
RAMP_WINDOW, SPIKE_WINDOW = (165.0, 585.0), (585.0, 645.0)
SECOND_REPLICA_MAX_RPS = 8.5
GPU_SHARE_MAX, GOODPUT_MIN, GOODPUT_MAX_GAP = 0.75, 0.95, 0.03
START_MARGIN_S = 15.0
SLOT1_MIN_SHARE, SLOT1_WINDOW_S = 0.30, 60.0
TPOT_BUCKET_S, TPOT_BUCKET_MIN = 30.0, 5
USD_PER_GPU_HOUR = 3.03
CAP_INTERACTIVE = 96
LAG_WARN_MS = 100.0


@dataclass(frozen=True)
class Verdict:
    rule: str
    status: str
    detail: str

    def line(self) -> str:
        return f"{self.rule} {self.status}: {self.detail}"


@dataclass(frozen=True)
class Cell:
    name: str
    requests: list[dict]
    ticks: list[dict]
    decisions: list[dict]
    samples: list[dict]
    events: list[dict]
    load: dict
    env: dict

    @property
    def scale(self) -> float:
        return float(self.load["time_scale"])

    @property
    def s(self) -> float:
        return float(self.env["S"])

    def to_wall(self, trace_t: float) -> float:
        return self.load["t0"] + (trace_t - self.load["start"]) / self.scale

    def to_trace(self, wall: float) -> float:
        return (wall - self.load["t0"]) * self.scale + self.load["start"]


def parse_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def parse_samples(text: str) -> list[dict]:
    lines = text.splitlines()
    if not lines:
        return []
    columns = lines[0].split("\t")
    rows = [dict(zip(columns, line.split("\t"))) for line in lines[1:]]
    return [{**r, "t": float(r["t"]), "waiting": int(r["waiting"]), "running": int(r["running"])}
            for r in rows]


def percentile(values: list[float], q: float) -> float | None:
    """Linear interpolation, q in [0, 100]."""
    if not values:
        return None
    ordered = sorted(values)
    at = (len(ordered) - 1) * q / 100
    low, frac = int(at), at - int(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * frac


def of_class(requests: list[dict], cls: str) -> list[dict]:
    return [r for r in requests if r["class"] == cls]


def in_window(requests: list[dict], window: tuple[float, float]) -> list[dict]:
    return [r for r in requests if window[0] <= r["trace_t"] < window[1]]


def wait_ms(r: dict) -> float | None:
    """TTFT at the client. It starts at the send, so it includes any time held by the router."""
    return None if r["t_first"] is None else (r["t_first"] - r["t_sent"]) * 1000


def is_good(r: dict) -> bool:
    if r["status"] != 200 or r["truncated"] or r["error"]:
        return False
    ttft = wait_ms(r)
    return ttft is not None and ttft <= TTFT_LIMIT_MS \
        and (r["tpot_ms"] is None or r["tpot_ms"] <= TPOT_LIMIT_MS)


def goodput(requests: list[dict]) -> float | None:
    return sum(map(is_good, requests)) / len(requests) if requests else None


def tpot_p90(requests: list[dict]) -> float | None:
    return percentile([r["tpot_ms"] for r in requests if r["tpot_ms"] is not None], 90)


def pct(share: float | None) -> str:
    return "n/a" if share is None else f"{share * 100:.1f}%"


def num(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def split_cold(requests: list[dict], ticks: list[dict]) -> tuple[list[dict], list[dict]]:
    """(warm, cold): cold requests arrived while no slot was READY."""
    times = [x["t"] for x in ticks]
    warm, cold = [], []
    for r in requests:
        i = bisect_right(times, r["t_scheduled"])
        ready = i > 0 and "ready" in ticks[i - 1]["phases"]
        (warm if ready else cold).append(r)
    return warm, cold


def gpu_seconds(ticks: list[dict], t0: float, t1: float) -> float:
    """Slot-seconds from STARTING to OFF in [t0, t1]. Each tick's phases hold until the next tick."""
    total = 0.0
    for a, b in pairwise(ticks):
        overlap = min(b["t"], t1) - max(a["t"], t0)
        if overlap > 0:
            total += overlap * sum(p != "off" for p in a["phases"])
    return total


def first_tick(ticks: list[dict], after: float, test) -> dict | None:
    return next((x for x in ticks if x["t"] >= after and test(x)), None)


def backlog_clear(ticks: list[dict]) -> float | None:
    """Wall time from when the router stopped holding interactive work after the first READY."""
    ready = first_tick(ticks, -1.0, lambda x: "ready" in x["phases"])
    if ready is None:
        return None
    clear = first_tick(ticks, ready["t"], lambda x: 0 < x["phases"].count("ready")
                       and x["signal"]["interactive"] <= CAP_INTERACTIVE * x["phases"].count("ready"))
    return clear["t"] if clear else None


def not_evaluated(rule: str, *missing: str) -> Verdict:
    return Verdict(rule, NOT_EVALUATED, f"needs {' and '.join(missing)}")


def p1(loop: Cell | None, name: str = "C1") -> Verdict:
    if loop is None:
        return not_evaluated("P1", name)
    tick = first_tick(loop.ticks, -1.0, lambda x: x["serve"] >= 2)
    if tick is None:
        return Verdict("P1", FAIL, "the second replica was never requested")
    phases = json.loads(TRACE_FILE.read_text())["main"]["phases"]
    rps = rate_at(phases, loop.to_trace(tick["t"]))
    status = PASS if rps < SECOND_REPLICA_MAX_RPS else FAIL
    return Verdict("P1", status, f"requested at {rps:.2f} req/s, t={loop.to_trace(tick['t']):.0f}; "
                                 f"limit {SECOND_REPLICA_MAX_RPS}")


def p2(loop: Cell | None, base: Cell | None, names: tuple[str, str] = ("C1", "C2")) -> Verdict:
    if loop is None or base is None:
        return not_evaluated("P2", *names)
    warm, cold = split_cold(of_class(loop.requests, "interactive"), loop.ticks)
    clear = backlog_clear(loop.ticks)
    start = RAMP_WINDOW[0] if clear is None else max(RAMP_WINDOW[0], loop.to_trace(clear))
    window = (start, RAMP_WINDOW[1])
    ours = goodput(in_window(warm, window))
    theirs = goodput(in_window(of_class(base.requests, "interactive"), window))
    if ours is None or theirs is None:
        return Verdict("P2", FAIL, "no interactive request in the window")
    gap = theirs - ours
    status = PASS if ours >= GOODPUT_MIN and gap <= GOODPUT_MAX_GAP else FAIL
    return Verdict("P2", status, f"goodput t={start:.0f} to 585 {pct(ours)}, baseline {pct(theirs)}, "
                                 f"gap {gap * 100:.1f} points; from t=165 with the cold backlog "
                                 f"{pct(goodput(in_window(warm, RAMP_WINDOW)))}, "
                                 f"{len(in_window(cold, RAMP_WINDOW))} cold-window requests apart")


def p3(loop: Cell | None, name: str = "C1") -> Verdict:
    if loop is None:
        return not_evaluated("P3", name)
    t0, t1 = loop.to_wall(0.0), loop.to_wall(TRACE_S)
    held, static = gpu_seconds(loop.ticks, t0, t1), 2 * (t1 - t0)
    share = held / static
    saved_usd = (static - held) * loop.scale / 3600 * USD_PER_GPU_HOUR
    return Verdict("P3", PASS if share <= GPU_SHARE_MAX else FAIL,
                   f"{held:.0f} of {static:.0f} GPU-seconds, {share:.3f}; limit {GPU_SHARE_MAX}; "
                   f"${saved_usd:.2f} of engine time at ${USD_PER_GPU_HOUR}/GPU-hour")


def p4(c2: Cell | None, c4: Cell | None) -> Verdict:
    if c2 is None or c4 is None:
        return not_evaluated("P4", "C2", "C4")
    gated = tpot_p90(in_window(of_class(c2.requests, "interactive"), SPIKE_WINDOW))
    blind = tpot_p90(in_window(of_class(c4.requests, "interactive"), SPIKE_WINDOW))
    if gated is None or blind is None:
        return Verdict("P4", FAIL, "no TPOT in the spike window")
    ok = gated <= TPOT_LIMIT_MS < blind
    return Verdict("P4", PASS if ok else FAIL,
                   f"spike p90 TPOT C2 {gated:.1f} ms, C4 {blind:.1f} ms; limit {TPOT_LIMIT_MS}")


def p5(loop: Cell | None, name: str = "C1") -> Verdict:
    if loop is None:
        return not_evaluated("P5", name)
    _, cold = split_cold(loop.requests, loop.ticks)
    if not cold:
        return Verdict("P5", FAIL, "no request arrived in a cold window")
    failed = sum(r["status"] != 200 or r["truncated"] or bool(r["error"]) for r in cold)
    waits = [w for w in map(wait_ms, of_class(cold, "interactive")) if w is not None]
    most = max(waits, default=0.0) / 1000
    limit = loop.s + START_MARGIN_S
    return Verdict("P5", PASS if failed == 0 and most <= limit else FAIL,
                   f"{len(cold) - failed} of {len(cold)} completed, "
                   f"maximum interactive wait {most:.1f} s; limit {limit:.1f} s")


def tpot_buckets(c3: Cell) -> dict[int, float]:
    """p90 TPOT by arrival, per bucket of trace time, where a bucket has enough requests."""
    buckets: dict[int, list[float]] = {}
    for r in of_class(c3.requests, "interactive"):
        if r["tpot_ms"] is not None:
            buckets.setdefault(int(r["trace_t"] // TPOT_BUCKET_S), []).append(r["tpot_ms"])
    return {k: percentile(v, 90) for k, v in buckets.items() if len(v) >= TPOT_BUCKET_MIN}


def p6(c3: Cell | None) -> Verdict:
    if c3 is None:
        return not_evaluated("P6", "C3")
    slow = sorted(k for k, v in tpot_buckets(c3).items() if v > TPOT_LIMIT_MS)
    if not slow:
        return Verdict("P6", FAIL, "TPOT p90 never exceeded the limit, so the rule did not fire")
    busy = sorted(c3.to_trace(x["t"]) for x in c3.samples if x["waiting"] > 0)
    hits = [k for k in slow if any(k * TPOT_BUCKET_S <= t < (k + 1) * TPOT_BUCKET_S for t in busy)]
    first_slow = slow[0] * TPOT_BUCKET_S
    first_busy = f"t={busy[0]:.0f}" if busy else "never"
    return Verdict("P6", FAIL if hits else PASS,
                   f"{len(slow)} of {len(tpot_buckets(c3))} {TPOT_BUCKET_S:.0f} s buckets over "
                   f"{TPOT_LIMIT_MS} ms, {len(hits)} of them with waiting above 0; first slow "
                   f"t={first_slow:.0f}, first waiting {first_busy}")


def p7(c5: Cell | None) -> Verdict:
    if c5 is None:
        return not_evaluated("P7", "C5")
    reclaim = next((e for e in c5.events if e["event"] == "reclaim"), None)
    if reclaim is None:
        return Verdict("P7", FAIL, "no reclaim event recorded")
    slot = int(reclaim["args"].split()[0])
    batch_lost = sum(r["status"] != 200 for r in of_class(c5.requests, "batch"))
    interactive = of_class(c5.requests, "interactive")
    unstarted_lost = sum(r["status"] != 200 for r in interactive)
    truncated = sum(r["truncated"] for r in interactive)
    gone = first_tick(c5.ticks, reclaim["t"], lambda x: x["phases"][slot] in ("off", "starting"))
    back = gone and first_tick(c5.ticks, gone["t"], lambda x: x["phases"][slot] == "ready")
    counts = f"batch lost {batch_lost}, unstarted interactive lost {unstarted_lost}, " \
             f"truncated {truncated}"
    if not back:
        return Verdict("P7", FAIL, f"{counts}; slot {slot} did not come back")
    limit = c5.s + START_MARGIN_S
    from_gone, from_reclaim = back["t"] - gone["t"], back["t"] - reclaim["t"]
    ok = batch_lost == 0 and unstarted_lost == 0 and from_gone <= limit
    return Verdict("P7", PASS if ok else FAIL,
                   f"{counts}; slot back {from_gone:.1f} s after the engine exited, "
                   f"{from_reclaim:.1f} s after the reclaim; limit {limit:.1f} s")


def p8(loop: Cell | None, name: str = "C1") -> Verdict:
    if loop is None:
        return not_evaluated("P8", name)
    ready = first_tick(loop.ticks, -1.0, lambda x: x["phases"][1] == "ready")
    if ready is None:
        return Verdict("P8", FAIL, "slot 1 never became READY")
    end = ready["t"] + SLOT1_WINDOW_S / loop.scale
    chosen = [d["replica_index"] for d in loop.decisions
              if d["t_granted"] is not None and ready["t"] <= d["t_granted"] < end]
    if not chosen:
        return Verdict("P8", FAIL, "no request was granted after slot 1 became READY")
    share = chosen.count(1) / len(chosen)
    return Verdict("P8", PASS if share >= SLOT1_MIN_SHARE else FAIL,
                   f"slot 1 took {pct(share)} of {len(chosen)} requests in the "
                   f"{SLOT1_WINDOW_S:.0f} s after READY; limit {pct(SLOT1_MIN_SHARE)}")


def loop_rules(loop: Cell | None, base: Cell | None, names: tuple[str, str]) -> list[Verdict]:
    return [p1(loop, names[0]), p2(loop, base, names), p3(loop, names[0]), p5(loop, names[0]),
            p8(loop, names[0])]


def agreement(first: list[Verdict], repeat: list[Verdict]) -> list[str]:
    """C6 against C1: one line per rule, on the direction of the verdict."""
    lines = []
    for a, b in zip(first, repeat):
        if NOT_EVALUATED in (a.status, b.status):
            lines.append(f"{a.rule} {NOT_EVALUATED}: C1 {a.status}, C6 {b.status}")
        else:
            word = "AGREE" if a.status == b.status else "DISAGREE"
            lines.append(f"{a.rule} {word}: C1 {a.status}, C6 {b.status}")
    return lines


def cold_row(c: Cell) -> str:
    _, cold = split_cold(c.requests, c.ticks)
    interactive = of_class(cold, "interactive")
    waits = [w / 1000 for w in map(wait_ms, interactive) if w is not None]
    ok = sum(r["status"] == 200 and not r["truncated"] for r in cold)
    return (f"{c.name}: {len(interactive)} interactive and {len(cold) - len(interactive)} batch "
            f"arrived with no slot READY, {ok} of {len(cold)} completed, wait p50 "
            f"{num(percentile(waits, 50))} s, maximum {num(max(waits, default=None))} s")


def harness_row(c: Cell) -> str:
    shed = sum(r["status"] in (429, 503) for r in c.requests)
    errors = sum(bool(r["error"]) for r in c.requests)
    other = sum(r["status"] not in (200, 429, 503) and not r["error"] for r in c.requests)
    truncated = sum(r["truncated"] for r in c.requests)
    lag = c.load["lag_ms"]
    warn = "  LOAD GENERATOR LAGGED" if (lag["p99"] or 0) > LAG_WARN_MS else ""
    return (f"{c.name}: {len(c.requests)} requests, goodput "
            f"{pct(goodput(of_class(c.requests, 'interactive')))}, shed {shed}, client errors "
            f"{errors}, other status {other}, truncated {truncated}, lag p99 {num(lag['p99'])} ms, max {num(lag['max'])} ms{warn}")


def backlog_row(c: Cell) -> str:
    clear = backlog_clear(c.ticks)
    if clear is None:
        return f"{c.name}: interactive backlog never cleared"
    return f"{c.name}: backlog clear at t={c.to_trace(clear):.0f}"


def load_cell(path: Path) -> Cell:
    def read(name: str) -> str:
        f = path / name
        return f.read_text() if f.exists() else ""

    return Cell(path.name, parse_jsonl(read("requests.jsonl")), parse_jsonl(read("ticks.jsonl")),
                parse_jsonl(read("decisions.jsonl")), parse_samples(read("samples.tsv")),
                parse_jsonl(read("events.jsonl")), json.loads(read("requests.meta.json")),
                json.loads(read("env.json")))


def find_cell(run: Path, prefix: str) -> Cell | None:
    for path in sorted(run.glob(f"{prefix}-*")):
        if (path / "DONE").exists():
            return load_cell(path)
    return None


def report(cells: dict[str, Cell | None]) -> list[str]:
    c1, c2, c3, c4, c5, c6, c7 = (cells.get(f"c{i}") for i in range(1, 8))
    first = loop_rules(c1, c2, ("C1", "C2"))
    by_rule = {v.rule: v for v in [*first, p4(c2, c4), p6(c3), p7(c5)]}
    repeat = loop_rules(c6, c7 or c2, ("C6", "C7 or C2"))
    present = [c for c in cells.values() if c is not None]
    return [
        "Rules",
        *[by_rule[f"P{i}"].line() for i in range(1, 9)],
        "",
        "C6 against C1",
        *agreement(first, repeat),
        "",
        "Cold window",
        *[cold_row(c) for c in (c1, c6) if c is not None],
        "",
        "Backlog after the first cold start, where P2 starts",
        *[backlog_row(c) for c in (c1, c6) if c is not None],
        "",
        "Harness",
        *[harness_row(c) for c in present],
    ]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run = Path(sys.argv[1])
    print("\n".join(report({f"c{i}": find_cell(run, f"c{i}") for i in range(1, 8)})))


if __name__ == "__main__":
    main()
