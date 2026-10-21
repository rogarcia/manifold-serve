# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas>=2.2"]
# ///
"""Summarize bench/m1 results.

  uv run summarize.py ROOT [--warmup-s 60] [--out CSV]
      One row per finished cell under ROOT into summary.csv, and markdown headline tables.
  uv run summarize.py --calibrate ROOT [--slo-ms 1000] [--duration-s 420] [--knee N] [--write-knee FILE]
      Per load level: p90 TTFT and waiting growth. Picks the knee, sizes the matrix loads.
  uv run summarize.py --self-test
      Runs both on tests/fixture/ and checks the numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LOAD_FRACS = (30, 70, 95)


def read_prom(path: Path) -> dict[str, float]:
    """Metric name -> value summed over label sets."""
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
    return out


def delta(cell: Path, replica: str) -> dict[str, float]:
    before = read_prom(cell / f"metrics_before_{replica}.txt")
    after = read_prom(cell / f"metrics_after_{replica}.txt")
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}


def ratio(num: float, den: float) -> float:
    return num / den if den else math.nan


def pct(values: pd.Series, q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else math.nan


@dataclass
class Cell:
    run: str
    model_tag: str
    pressure: str
    workload: str
    policy: str
    clients: int
    seed: int
    load_label: str | None
    active_effective: int
    nreq: int
    requests: int
    decisions: int
    errors: int
    window_s: float
    achieved_rps: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p90_ms: float
    share_a: float
    share_b: float
    hit_rate_a: float
    hit_rate_b: float
    hit_rate: float
    spill_share: float
    queue_ms_mean: float
    prefill_ms_mean: float
    join_coverage: float
    same_replica_share: float
    waiting_mean: float
    waiting_slope_per_min: float
    waiting_first_q: float
    waiting_last_q: float


def load_stats(cell: Path, warmup_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All bench records, and the ones that started after the warm-up."""
    df = pd.DataFrame(json.loads((cell / "stats.json").read_text()))
    t0 = df["start_time_ms"].min()
    df["t_s"] = (df["start_time_ms"] - t0) / 1000.0
    df["end_s"] = df["t_s"] + df["latency_ms"] / 1000.0
    return df, df[df["t_s"] >= warmup_s]


def load_decisions(cell: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in (cell / "decisions.jsonl").read_text().splitlines() if line.strip()]
    d = pd.DataFrame(rows, columns=list(rows[0]) if rows else
                     ["session_id", "replica_index", "spilled", "status", "error", "t_received"])
    d["ok"] = (d["status"] == 200) & d["error"].isna()
    return d


def join_turns(stats: pd.DataFrame, dec: pd.DataFrame) -> tuple[float, float]:
    """Pair bench records with successful decisions by session and order within the session.

    A session whose counts differ is left unpaired. Returns (share of bench records paired,
    share of follow-up turns sent to the replica that served the previous turn)."""
    ok = dec[dec["ok"]].sort_values("t_received")
    by_session = {s: g["replica_index"].tolist() for s, g in ok.groupby("session_id")}
    paired = same = followups = 0
    for conv, g in stats.sort_values("start_time_ms").groupby("conversation_id"):
        replicas = by_session.get(str(conv))
        if replicas is None or len(replicas) != len(g):
            continue
        paired += len(g)
        for prev, cur in zip(replicas, replicas[1:]):
            followups += 1
            same += prev == cur
    return ratio(paired, len(stats)), ratio(same, followups)


def waiting_growth(cell: Path, t_from: float) -> tuple[float, float, float, float]:
    """Total waiting across replicas per tick after t_from (Unix s): mean, slope per minute,
    and the means of the first and last quarter of the window."""
    p = cell / "samples.tsv"
    if not p.exists():
        return (math.nan,) * 4
    s = pd.read_csv(p, sep="\t")
    s = s[s["t"] >= t_from]
    tot = s.groupby("t")["waiting"].sum()
    if len(tot) < 4:
        return (math.nan,) * 4
    t = tot.index.to_numpy() - tot.index[0]
    slope = float(np.polyfit(t, tot.to_numpy(), 1)[0]) * 60.0
    q = max(1, len(tot) // 4)
    return float(tot.mean()), slope, float(tot.iloc[:q].mean()), float(tot.iloc[-q:].mean())


def summarize_cell(cell: Path, warmup_s: float) -> Cell:
    meta = json.loads((cell / "meta.json").read_text())
    stats, kept = load_stats(cell, warmup_s)
    dec = load_decisions(cell)
    ok = dec[dec["ok"]]
    window_s = float(kept["end_s"].max() - warmup_s) if len(kept) else math.nan
    da, db = delta(cell, "a"), delta(cell, "b")
    hits, queries = "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"
    q_sum, q_cnt = "vllm:request_queue_time_seconds_sum", "vllm:request_queue_time_seconds_count"
    p_sum, p_cnt = "vllm:request_prefill_time_seconds_sum", "vllm:request_prefill_time_seconds_count"
    both = {k: da.get(k, 0.0) + db.get(k, 0.0) for k in set(da) | set(db)}
    coverage, same = join_turns(stats, dec)
    w_mean, w_slope, w_first, w_last = waiting_growth(cell, meta["t_start"] + warmup_s)
    return Cell(
        run=meta["run"], model_tag=meta["model_tag"], pressure=meta["pressure"],
        workload=meta["workload"], policy=meta["policy"], clients=meta["clients"],
        seed=meta["seed"], load_label=meta.get("load_label"),
        active_effective=meta["active_effective"], nreq=meta["nreq"],
        requests=len(stats), decisions=len(dec), errors=int((~dec["ok"]).sum()),
        window_s=window_s, achieved_rps=ratio(len(kept), window_s),
        ttft_p50_ms=pct(kept["ttft_ms"], 50), ttft_p90_ms=pct(kept["ttft_ms"], 90),
        ttft_p99_ms=pct(kept["ttft_ms"], 99),
        tpot_p50_ms=pct(kept["tpot_ms"], 50), tpot_p90_ms=pct(kept["tpot_ms"], 90),
        share_a=ratio((ok["replica_index"] == 0).sum(), len(ok)),
        share_b=ratio((ok["replica_index"] == 1).sum(), len(ok)),
        hit_rate_a=ratio(da.get(hits, 0.0), da.get(queries, 0.0)),
        hit_rate_b=ratio(db.get(hits, 0.0), db.get(queries, 0.0)),
        hit_rate=ratio(both.get(hits, 0.0), both.get(queries, 0.0)),
        spill_share=ratio(ok["spilled"].sum(), len(ok)),
        queue_ms_mean=1000 * ratio(both.get(q_sum, 0.0), both.get(q_cnt, 0.0)),
        prefill_ms_mean=1000 * ratio(both.get(p_sum, 0.0), both.get(p_cnt, 0.0)),
        join_coverage=coverage, same_replica_share=same,
        waiting_mean=w_mean, waiting_slope_per_min=w_slope,
        waiting_first_q=w_first, waiting_last_q=w_last,
    )


def summarize_tree(root: Path, warmup_s: float) -> pd.DataFrame:
    cells = sorted(p.parent for p in root.rglob("DONE") if (p.parent / "meta.json").exists())
    return pd.DataFrame([asdict(summarize_cell(c, warmup_s)) for c in cells])


def spread(s: pd.Series) -> str:
    if len(s) == 1:
        return f"{s.iloc[0]:.0f}"
    return f"{s.mean():.0f} ({s.min():.0f}-{s.max():.0f})"


def headline(df: pd.DataFrame) -> str:
    out = []
    df = df.assign(load=df["load_label"].fillna("c" + df["clients"].astype(str)))
    for (model, workload), g in df.groupby(["model_tag", "workload"]):
        out += [f"\n### {model}, {workload}\n",
                "| pressure | load | policy | seeds | TTFT p50 ms | TTFT p90 ms | hit rate | share A | spill | errors |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for (pressure, load, policy), c in g.groupby(["pressure", "load", "policy"]):
            out.append(
                f"| {pressure} | {load} | {policy} | {len(c)} | {spread(c['ttft_p50_ms'])} | "
                f"{spread(c['ttft_p90_ms'])} | {c['hit_rate'].mean():.2f} | {c['share_a'].mean():.2f} | "
                f"{c['spill_share'].mean():.2f} | {int(c['errors'].sum())} |")
    return "\n".join(out)


def calibrate(df: pd.DataFrame, slo_ms: float, duration_s: float, knee: int | None) -> tuple[str, dict]:
    df = df[(df["policy"] == "least_load") & (df["workload"] == "sharegpt")].sort_values("clients")
    df = df.assign(
        over_slo=df["ttft_p90_ms"] > slo_ms,
        # "Grows without bound" in practice: a positive trend and a last quarter clearly above the first.
        waiting_grows=(df["waiting_slope_per_min"] > 0.5) & (df["waiting_last_q"] > df["waiting_first_q"] + 2),
    )
    lines = ["| clients | active | req/s | TTFT p50 ms | TTFT p90 ms | waiting mean | waiting slope /min | over SLO | waiting grows |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.clients} | {r.active_effective} | {r.achieved_rps:.2f} | {r.ttft_p50_ms:.0f} | "
                     f"{r.ttft_p90_ms:.0f} | {r.waiting_mean:.1f} | {r.waiting_slope_per_min:.2f} | "
                     f"{'yes' if r.over_slo else 'no'} | {'yes' if r.waiting_grows else 'no'} |")
    failing = df[df["over_slo"] | df["waiting_grows"]]
    suggested = int(failing["clients"].iloc[0]) if len(failing) else None
    lines.append("")
    lines.append(f"knee by the rule (lowest level over the {slo_ms:.0f} ms p90 SLO or with growing waiting): "
                 f"{suggested if suggested is not None else 'none reached; extend CALIB_CLIENTS or pass --knee'}")
    knee = knee or suggested
    if knee is None:
        return "\n".join(lines), {}
    env = {"KNEE_CLIENTS": knee}
    for f in LOAD_FRACS:
        c = max(1, round(knee * f / 100))
        rps = float(np.interp(c, df["clients"], df["achieved_rps"]))
        env[f"CLIENTS_{f}"] = c
        env[f"NREQ_{f}"] = max(4 * c, round(duration_s * rps))
        lines.append(f"load {f}%: {c} clients, about {rps:.2f} req/s, NREQ {env[f'NREQ_{f}']} for {duration_s:.0f} s")
    if len({env[f"CLIENTS_{f}"] for f in LOAD_FRACS}) < len(LOAD_FRACS):
        lines.append("WARNING: two load levels round to the same client count")
    return "\n".join(lines), env


def write_env(path: Path, env: dict, note: str) -> None:
    path.write_text(f"# {note}\n" + "".join(f"{k}={v}\n" for k, v in env.items()))


def self_test() -> None:
    fx = HERE / "tests" / "fixture"
    df = summarize_tree(fx / "matrix", warmup_s=1.0)
    assert len(df) == 3, f"expected 3 cells, got {len(df)}"
    c = df[(df["policy"] == "prefix_aware") & (df["seed"] == 1)].iloc[0]
    want = {
        "requests": 6, "decisions": 7, "errors": 1,
        "window_s": 4.5, "achieved_rps": 4 / 4.5,
        "ttft_p50_ms": 250.0, "ttft_p90_ms": 370.0, "ttft_p99_ms": 397.0,
        "tpot_p50_ms": 25.0, "tpot_p90_ms": 37.0,
        "share_a": 4 / 6, "share_b": 2 / 6,
        "hit_rate_a": 0.5, "hit_rate_b": 0.1, "hit_rate": 0.3,
        "spill_share": 1 / 6, "queue_ms_mean": 75.0, "prefill_ms_mean": 30.0,
        "join_coverage": 1.0, "same_replica_share": 0.75,
    }
    for k, v in want.items():
        assert math.isclose(c[k], v, rel_tol=1e-9), f"{k}: got {c[k]}, want {v}"
    table = headline(df)
    assert "| pressure | 70 | prefix_aware | 2 | 262 (250-275) |" in table, table
    cal = summarize_tree(fx / "calib", warmup_s=1.0)
    text, env = calibrate(cal, slo_ms=1000, duration_s=100, knee=None)
    # c4 is under the SLO but its waiting count climbs, so the knee is 4, not 8.
    assert env["KNEE_CLIENTS"] == 4, text
    assert (env["CLIENTS_30"], env["CLIENTS_70"], env["CLIENTS_95"]) == (1, 3, 4), env
    assert env["NREQ_70"] == 150, env
    print(table)
    print(text)
    print("self-test passed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", type=Path)
    ap.add_argument("--warmup-s", type=float, default=60.0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--slo-ms", type=float, default=1000.0)
    ap.add_argument("--duration-s", type=float, default=420.0)
    ap.add_argument("--knee", type=int)
    ap.add_argument("--write-knee", type=Path)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.root is None:
        ap.error("ROOT is required")
    df = summarize_tree(a.root, a.warmup_s)
    if df.empty:
        sys.exit(f"no finished cells under {a.root}")
    out = a.out or a.root / "summary.csv"
    df.to_csv(out, index=False)
    print(f"{len(df)} cells -> {out}")
    if a.calibrate:
        text, env = calibrate(df, a.slo_ms, a.duration_s, a.knee)
        print(text)
        if a.write_knee and env:
            write_env(a.write_knee, env, f"summarize.py --calibrate {a.root}, SLO {a.slo_ms:.0f} ms, "
                      f"{a.duration_s:.0f} s per cell{', knee set by hand' if a.knee else ''}")
            print(f"wrote {a.write_knee}")
    else:
        print(headline(df))


if __name__ == "__main__":
    main()
