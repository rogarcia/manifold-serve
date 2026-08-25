# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Render the M0-part2 A/B charts.  Run:  uv run bench/m0-part2/charts.py

Compares three variants on the same 1x RTX 5090 / Qwen3.8-27B-NVFP4 / loadgen protocol:
  m0      = marlin GEMM + Triton attention fallbacks (bench/m0/rate_*.csv, 2026-08-22)
  native  = FlashInfer CUTLASS NVFP4 + fp8-KV FlashInfer attention (bench/m0-part2/rate_*.csv)
  mtp     = native + MTP-3 speculative decoding (bench/m0-part2/mtp/rate_*.csv)

Outputs bench/m0-part2/charts/*.svg and *.png (SVG for the blog, PNG for X/LinkedIn).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
OUT = HERE / "charts"
OUT.mkdir(exist_ok=True)
RATES = [0.5, 1, 2, 4, 8]

# validated categorical triple (dataviz reference palette slots 1-3); aqua is below
# 3:1 on white so every aqua mark carries a direct label
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1c2026", "#5c6673", "#e3e6e2"
VARIANTS = [  # (key, dir, color, label) — color follows the variant everywhere
    ("m0", HERE.parent / "m0", BLUE, "M0 fallback (marlin + Triton)"),
    ("native", HERE, ORANGE, "native NVFP4 (FlashInfer, fp8 KV)"),
    ("mtp", HERE / "mtp", AQUA, "native + MTP-3"),
]

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
        "axes.axisbelow": True,
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


def load(dirpath: Path, rate: float) -> list[dict]:
    with open(dirpath / f"rate_{rate}.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "200" and r["ttft_ms"]]
    for r in rows:
        r["ttft"] = float(r["ttft_ms"])
        r["total"] = float(r["total_ms"])
        r["tok"] = int(r["output_tokens"])
        r["start"] = float(r["start"])
        r["tpot"] = (r["total"] - r["ttft"]) / (r["tok"] - 1) if r["tok"] > 1 else None
    return rows


def pct(v: list[float], q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


stats: dict[str, dict[str, list[float]]] = {}
for key, d, _, _ in VARIANTS:
    data = {r: load(d, r) for r in RATES}
    thr = []
    for r in RATES:
        rows = data[r]
        span = max(x["start"] + x["total"] / 1000 for x in rows) - min(x["start"] for x in rows)
        thr.append(sum(x["tok"] for x in rows) / span)
    stats[key] = {
        "ttft50": [pct([x["ttft"] for x in data[r]], 0.5) for r in RATES],
        "tpot50": [pct([x["tpot"] for x in data[r] if x["tpot"]], 0.5) for r in RATES],
        "thr": thr,
    }

# from the pods' startup logs (vllm-startup.log here and in mtp/; M0 numbers in bench/m0).
# NB: all three runs use fp8 KV (M0 too — see its non-default args); tok/GiB is ~25.3k in
# m0 and native alike, so the pool delta is budget (util 0.92->0.95 + lower non-KV usage),
# not bytes-per-token
KV_TOKENS = {"m0": 53_399, "native": 92_235, "mtp": 44_600}
KV_CONC = {"m0": "1.63×", "native": "2.81×", "mtp": "1.36×"}
# spec-decode counters read live on the pod during the MTP sweep (bench/m0-part2/runs.md);
# the saved metrics_*.txt were grep-filtered too narrowly and miss these lines
MTP_ACCEPT_PER_POS = [0.61, 0.34, 0.18]
MTP_ACCEPT_OVERALL = 50_355 / 134_049  # 37.6%
MTP_RECIPE_CLAIM = 0.75


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", OUT / name)


def rate_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(RATES, [str(r) for r in RATES])
    ax.set_xlabel("arrival rate (req/s, Poisson)")


# 1. Saturated throughput A/B: the headline
fig, ax = plt.subplots(figsize=(7, 4.2))
for key, _, color, label in VARIANTS:
    ax.plot(RATES, stats[key]["thr"], "-o", color=color, lw=2, ms=6, label=label)
    ax.annotate(f"{stats[key]['thr'][-1]:.0f}", (RATES[-1], stats[key]["thr"][-1]),
                xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
rate_axis(ax)
ax.set_ylim(0, 200)
ax.set_ylabel("output tokens / s (aggregate)")
ax.set_title("Same GPU, same model: native kernels +43% saturated throughput,\nMTP-3 collapses to 28 tok/s")
ax.legend(frameon=False, loc="upper left")
save(fig, "throughput_ab")

# 2. TTFT vs rate A/B: where the knee moved
fig, ax = plt.subplots(figsize=(7, 4.2))
for key, _, color, label in VARIANTS:
    ax.plot(RATES, stats[key]["ttft50"], "-o", color=color, lw=2, ms=6, label=label)
last = stats["mtp"]["ttft50"][-1]
ax.annotate(f"{last/1000:.0f} s", (RATES[-1], last), xytext=(0, 8),
            textcoords="offset points", ha="center", fontsize=9, color=INK)
for key in ("m0", "native"):
    v = stats[key]["ttft50"][2]
    ax.annotate(f"{v/1000:.1f} s" if v > 1000 else f"{v:.0f} ms", (2, v), xytext=(0, -16),
                textcoords="offset points", ha="center", fontsize=9, color=INK)
rate_axis(ax)
ax.set_yscale("log")
ax.set_ylabel("median time to first token (ms)")
ax.set_title("The knee moves right: at 2 req/s the fallback stack queues 7.8 s,\nnative answers in 287 ms")
ax.legend(frameon=False, loc="upper left")
save(fig, "ttft_ab")

# 3. The trade in two panels: batch-1 latency lost, KV headroom gained
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
names = ["M0\nfallback", "native", "native\n+MTP-3"]
colors = [c for _, _, c, _ in VARIANTS]
tpot = [stats[k]["tpot50"][0] for k, _, _, _ in VARIANTS]
a1.bar(names, tpot, color=colors, width=0.62)
a1.set_ylim(0, 130)
a1.grid(axis="x", visible=False)
a1.set_ylabel("TPOT p50 at 0.5 req/s (ms)")
a1.set_title("Marlin still wins batch-1 decode…")
for i, v in enumerate(tpot):
    a1.annotate(f"{v:.0f} ms", (i, v), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=9, color=INK)
kv = [KV_TOKENS[k] / 1000 for k, _, _, _ in VARIANTS]
a2.bar(names, kv, color=colors, width=0.62)
a2.set_ylim(0, 110)
a2.grid(axis="x", visible=False)
a2.set_ylabel("KV cache pool (k tokens @ 32k ctx)")
a2.set_title("…but the native config packs 1.7× the KV pool")
for i, (k, _, _, _) in enumerate(VARIANTS):
    a2.annotate(f"{KV_TOKENS[k]:,}\n({KV_CONC[k]})", (i, kv[i]), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=9, color=INK)
fig.suptitle("The A/B trade: ~24 ms/token slower at batch 1, 1.7× the KV pool",
             x=0.01, ha="left", fontweight="semibold", color=INK)
fig.tight_layout()
save(fig, "latency_vs_headroom")

# 4. MTP acceptance: the negative result
fig, ax = plt.subplots(figsize=(7, 3.8))
pos = ["draft token 1", "draft token 2", "draft token 3"]
ax.bar(pos, MTP_ACCEPT_PER_POS, color=AQUA, width=0.55)
ax.axhline(MTP_RECIPE_CLAIM, color=MUTED, lw=1.4, ls=(0, (4, 3)))
ax.text(0.99, MTP_RECIPE_CLAIM + 0.02, "recipe's reported overall acceptance (0.75)",
        transform=ax.get_yaxis_transform(), ha="right", fontsize=9, color=MUTED)
ax.axhline(MTP_ACCEPT_OVERALL, color=INK, lw=1.4)
ax.text(0.99, MTP_ACCEPT_OVERALL + 0.02,
        f"measured overall: {MTP_ACCEPT_OVERALL:.1%} (50,355 / 134,049 drafts)",
        transform=ax.get_yaxis_transform(), ha="right", fontsize=9, color=INK)
ax.set_ylim(0, 1)
ax.grid(axis="x", visible=False)
ax.set_ylabel("acceptance rate")
ax.set_title("MTP-3 acceptance by draft position, real chat traffic")
for i, v in enumerate(MTP_ACCEPT_PER_POS):
    ax.annotate(f"{v:.2f}", (i, v), xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=9, color="white", fontweight="semibold")
save(fig, "mtp_acceptance")

for key, _, _, _ in VARIANTS:
    print(key, {m: [round(v, 1) for v in vals] for m, vals in stats[key].items()})
