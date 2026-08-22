# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Render the M0 charts from bench/m0/*.csv.  Run:  uv run bench/m0/charts.py

Outputs bench/m0/charts/*.svg and *.png (SVG for the artifact/blog, PNG for X/LinkedIn).
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "charts"
OUT.mkdir(exist_ok=True)
RATES = [0.5, 1, 2, 4, 8]

# validated categorical pair (dataviz reference palette slots 1 and 2)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1c2026", "#5c6673", "#e3e6e2"
plt.rcParams.update(
    {
        "font.family": ["IBM Plex Sans", "Helvetica Neue", "Arial", "sans-serif"],
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.titlecolor": INK,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "figure.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    }
)


def load(rate: float) -> list[dict]:
    with open(HERE / f"rate_{rate}.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "200" and r["ttft_ms"]]
    for r in rows:
        r["ttft"] = float(r["ttft_ms"])
        r["total"] = float(r["total_ms"])
        r["tok"] = int(r["output_tokens"])
        r["turn"] = int(r["turn"])
        r["start"] = float(r["start"])
        r["tpot"] = (r["total"] - r["ttft"]) / (r["tok"] - 1) if r["tok"] > 1 else None
    return rows


def pct(v: list[float], q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


data = {r: load(r) for r in RATES}
ttft50 = [pct([x["ttft"] for x in data[r]], 0.5) for r in RATES]
ttft99 = [pct([x["ttft"] for x in data[r]], 0.99) for r in RATES]
tpot50 = [pct([x["tpot"] for x in data[r] if x["tpot"]], 0.5) for r in RATES]
tpot99 = [pct([x["tpot"] for x in data[r] if x["tpot"]], 0.99) for r in RATES]
thr = []
for r in RATES:
    rows = data[r]
    span = max(x["start"] + x["total"] / 1000 for x in rows) - min(x["start"] for x in rows)
    thr.append(sum(x["tok"] for x in rows) / span)


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", OUT / name)


# 1. TTFT vs arrival rate (log y): the knee
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(RATES, ttft50, "-o", color=BLUE, lw=2, ms=6, label="p50")
ax.plot(RATES, ttft99, "-o", color=ORANGE, lw=2, ms=6, label="p99")
ax.set_yscale("log")
ax.set_xscale("log", base=2)
ax.set_xticks(RATES, [str(r) for r in RATES])
ax.set_xlabel("arrival rate (req/s, Poisson)")
ax.set_ylabel("time to first token (ms)")
ax.set_title("TTFT vs arrival rate — the knee sits between 1 and 2 req/s")
ax.axvspan(1.4, 2.2, color=GRID, alpha=0.5, lw=0)
ax.annotate("capacity ≈ 1.8 req/s", (1.8, 260), ha="center", color=MUTED, fontsize=9)
for r, v in zip(RATES, ttft50):
    ax.annotate(f"{v/1000:.1f} s" if v > 1000 else f"{v:.0f} ms", (r, v), xytext=(0, -14),
                textcoords="offset points", ha="center", fontsize=8, color=INK)
ax.legend(frameon=False)
save(fig, "ttft_vs_rate")

# 2. TPOT and throughput: continuous batching in two panels (one axis each)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
a1.plot(RATES, tpot50, "-o", color=BLUE, lw=2, ms=6, label="p50")
a1.plot(RATES, tpot99, "-o", color=ORANGE, lw=2, ms=6, label="p99")
a1.set_xscale("log", base=2)
a1.set_xticks(RATES, [str(r) for r in RATES])
a1.set_ylim(0, 130)
a1.set_xlabel("arrival rate (req/s)")
a1.set_ylabel("time per output token (ms)")
a1.set_title("Per-token decode barely degrades…")
a1.legend(frameon=False)
a2.plot(RATES, thr, "-o", color=BLUE, lw=2, ms=6)
a2.set_xscale("log", base=2)
a2.set_xticks(RATES, [str(r) for r in RATES])
a2.set_ylim(0, 140)
a2.set_xlabel("arrival rate (req/s)")
a2.set_ylabel("output tokens / s (aggregate)")
a2.set_title("…while aggregate throughput grows 4×")
for r, v in zip(RATES, thr):
    a2.annotate(f"{v:.0f}", (r, v), xytext=(0, 7), textcoords="offset points", ha="center",
                fontsize=8, color=INK)
fig.suptitle("Continuous batching, measured", x=0.01, ha="left", fontweight="semibold", color=INK)
fig.tight_layout()
save(fig, "tpot_throughput")

# 3. TTFT by session turn at rate 1 (prefix cache should help turn N; it doesn't)
rows = data[1]
turns = sorted({x["turn"] for x in rows})
med = [statistics.median([x["ttft"] for x in rows if x["turn"] == t]) for t in turns]
n = [sum(1 for x in rows if x["turn"] == t) for t in turns]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar([str(t + 1) for t in turns], med, color=BLUE, width=0.7)
ax.set_ylim(0, max(med) * 1.3)
ax.grid(axis="x", visible=False)
ax.set_xlabel("turn within a multi-turn session (prompt grows each turn)")
ax.set_ylabel("median TTFT (ms)")
ax.set_title("TTFT by turn at 1 req/s — no prefix-cache benefit (0 hits all sweep)")
for i, (m, k) in enumerate(zip(med, n)):
    ax.annotate(f"{m:.0f}\n(n={k})", (i, m), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8, color=INK)
save(fig, "ttft_by_turn")

print({"ttft50": ttft50, "ttft99": ttft99, "tpot50": tpot50, "thr": thr, "turn_med": med})
