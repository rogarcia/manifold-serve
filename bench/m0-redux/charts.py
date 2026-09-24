# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Render the M0-redux charts.  Run:  uv run bench/m0-redux/charts.py

Reads every `results/<config>/<cell>-c<N>-greedy-thinkoff.json` written by `vllm bench serve`
(bench.sh) on one RTX PRO 6000 / Qwen3.8-27B-NVFP4 / vLLM 0.29.0, and draws the baseline-vs-MTP
comparison described in runs.md. KV-pool numbers come from the server startup logs (runs.md table).

Outputs bench/m0-redux/charts/*.svg and *.png (SVG for the blog, PNG for X/LinkedIn).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).parent
RES = HERE / "results"
OUT = HERE / "charts"
OUT.mkdir(exist_ok=True)

CONFIGS = ["baseline", "mtp-k1", "mtp-k2", "mtp-k3", "mtp-k5", "mtp-k7"]
K = {"baseline": 0, "mtp-k1": 1, "mtp-k2": 2, "mtp-k3": 3, "mtp-k5": 5, "mtp-k7": 7}
CELLS = [("code", "code (InstructCoder)"), ("chat", "chat (mt-bench)")]
CONCS = [1, 8]

# dataviz reference palette. Cells are identity -> categorical slots 1-2 (blue, orange).
# k is an ordered magnitude -> one-hue ordinal ramp (blue 250..650, validated with --ordinal).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, BASE = "#1c2026", "#5c6673", "#e3e6e2", "#8a929c"
CELL_COLOR = {"code": BLUE, "chat": ORANGE}
K_RAMP = {"baseline": BASE, "mtp-k1": "#86b6ef", "mtp-k2": "#5598e7", "mtp-k3": "#2a78d6",
          "mtp-k5": "#1c5cab", "mtp-k7": "#0d366b"}
LABEL = {"baseline": "baseline", "mtp-k1": "k=1", "mtp-k2": "k=2", "mtp-k3": "k=3",
         "mtp-k5": "k=5", "mtp-k7": "k=7"}

# GPU KV cache tokens per config at 262k max-model-len, fp8 KV, util 0.92 (runs.md, from
# the server startup logs; first value where a config was started twice)
KV_TOKENS = {"baseline": 2_018_810, "mtp-k1": 1_839_501, "mtp-k2": 1_808_197,
             "mtp-k3": 1_777_892, "mtp-k5": 1_705_360, "mtp-k7": 1_648_563}

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


def load() -> dict[tuple[str, str, int], dict]:
    """(config, cell, concurrency) -> result JSON. Missing cells (k=1/k=2 chat) are absent."""
    out = {}
    for cfg in CONFIGS:
        for cell, _ in CELLS:
            for c in CONCS:
                p = RES / cfg / f"{cell}-c{c}-greedy-thinkoff.json"
                if p.exists():
                    with open(p) as f:
                        out[(cfg, cell, c)] = json.load(f)
    return out


R = load()


def thr(cfg: str, cell: str, c: int) -> float:
    return R[(cfg, cell, c)]["output_throughput"]


def speedup(cfg: str, cell: str, c: int) -> float:
    return thr(cfg, cell, c) / thr("baseline", cell, c)


def have(cfg: str, cell: str, c: int) -> bool:
    return (cfg, cell, c) in R


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", OUT / name)


def cell_legend(ax, loc="upper left"):
    handles = [Line2D([], [], color=CELL_COLOR[k], lw=2, label=lbl) for k, lbl in CELLS]
    handles += [Line2D([], [], color=INK, lw=2, ls="-", label="concurrency 1"),
                Line2D([], [], color=INK, lw=2, ls=(0, (4, 2)), label="concurrency 8")]
    ax.legend(handles=handles, frameon=False, loc=loc, fontsize=9)


# 1. Headline: speedup vs k, per cell and concurrency
fig, ax = plt.subplots(figsize=(7, 4.2))
for cell, _ in CELLS:
    for c in CONCS:
        xs = [K[cfg] for cfg in CONFIGS if have(cfg, cell, c)]
        ys = [speedup(cfg, cell, c) for cfg in CONFIGS if have(cfg, cell, c)]
        ax.plot(xs, ys, "-o" if c == 1 else "--o", color=CELL_COLOR[cell], lw=2, ms=6,
                mec="white", mew=1)
        ax.annotate(f"{ys[-1]:.2f}×", (xs[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK)
ax.axhline(1, color=BASE, lw=1)
ax.set_xticks([0, 1, 2, 3, 5, 7], ["baseline", "1", "2", "3", "5", "7"])
ax.set_xlim(-0.3, 8.2)
ax.set_ylim(0.8, 3.8)
ax.set_xlabel("num_speculative_tokens (k)")
ax.set_ylabel("output tok/s relative to baseline")
ax.set_title("MTP speeds up every cell; the gain flattens past k=5\nand shrinks as the batch fills")
cell_legend(ax)
save(fig, "speedup_vs_k")

# 2. Absolute throughput, two panels (c=1, c=8), grouped bars per config
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=False)
w = 0.38
for ax, c in zip(axes, CONCS):
    xs = list(range(len(CONFIGS)))
    for j, (cell, _) in enumerate(CELLS):
        vals = [thr(cfg, cell, c) if have(cfg, cell, c) else 0 for cfg in CONFIGS]
        off = (j - 0.5) * (w + 0.04)
        ax.bar([x + off for x in xs], vals, width=w, color=CELL_COLOR[cell], edgecolor="white",
               linewidth=1)
        for x, v in zip(xs, vals):
            if v:
                ax.annotate(f"{v:.0f}", (x + off, v), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs, [LABEL[cfg] for cfg in CONFIGS])
    ax.grid(axis="x", visible=False)
    ax.set_title(f"concurrency {c}")
    ax.set_ylabel("output tokens / s")
axes[0].set_ylim(0, 230)
axes[1].set_ylim(0, 860)
axes[0].legend(handles=[Line2D([], [], color=CELL_COLOR[k], lw=6, label=l) for k, l in CELLS],
               frameon=False, loc="upper left", fontsize=9)
fig.suptitle("Output throughput by draft depth: k=1 and k=2 chat cells were not run",
             x=0.01, ha="left", fontweight="semibold", color=INK)
fig.tight_layout()
save(fig, "throughput_by_k")

# 3. Acceptance length vs speedup: where the rest went
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.plot([1, 7], [1, 7], color=BASE, lw=1, ls=(0, (4, 3)))
ax.text(6.05, 6.4, "speedup = acceptance length\n(free drafting)", fontsize=8.5, color=MUTED,
        ha="right")
for cell, _ in CELLS:
    for c in CONCS:
        pts = [(R[(cfg, cell, c)]["spec_decode_acceptance_length"], speedup(cfg, cell, c), cfg)
               for cfg in CONFIGS[1:] if have(cfg, cell, c)]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, "-o" if c == 1 else "--o", color=CELL_COLOR[cell], lw=1.6, ms=7,
                mec="white", mew=1)
        ax.annotate(f"c={c}", (xs[-1], ys[-1]), xytext=(7, -3), textcoords="offset points",
                    fontsize=8.5, color=INK)
for cfg in ["mtp-k1", "mtp-k3", "mtp-k5", "mtp-k7"]:
    a, s = R[(cfg, "code", 1)]["spec_decode_acceptance_length"], speedup(cfg, "code", 1)
    ax.annotate(LABEL[cfg], (a, s), xytext=(-4, 7), textcoords="offset points", ha="right",
                fontsize=8, color=MUTED)
ax.set_xlim(1, 7)
ax.set_ylim(1, 7)
ax.set_aspect("equal")
ax.set_xlabel("acceptance length (tokens emitted per decode step)")
ax.set_ylabel("speedup over baseline (output tok/s)")
ax.set_title("Accepted tokens are not free: code k=5 emits 5.2 tokens\nper step but runs 3.1× faster")
cell_legend(ax, loc="upper left")
save(fig, "acceptance_vs_speedup")

# 4. Per-position acceptance decay, k=5 and k=7, code vs chat (c=1; c=8 is identical)
fig, ax = plt.subplots(figsize=(7, 4))
for cell, _ in CELLS:
    for cfg, ls, mk in [("mtp-k5", "-", "o"), ("mtp-k7", (0, (4, 2)), "s")]:
        rates = R[(cfg, cell, 1)]["spec_decode_per_position_acceptance_rates"]
        xs = list(range(1, len(rates) + 1))
        ax.plot(xs, rates, ls=ls, marker=mk, color=CELL_COLOR[cell], lw=2, ms=6, mec="white",
                mew=1)
        ax.annotate(f"{cell} {LABEL[cfg]}", (xs[-1], rates[-1]), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
ax.set_xticks(range(1, 8))
ax.set_xlim(0.7, 8.4)
ax.set_ylim(0, 1)
ax.set_yticks([0, .25, .5, .75, 1], ["0", "25%", "50%", "75%", "100%"])
ax.set_xlabel("draft position")
ax.set_ylabel("acceptance rate at position")
ax.set_title("One MTP layer reused per position: acceptance decays with depth,\nfaster on chat than on code edits")
save(fig, "acceptance_by_position")

# 5. TPOT p50 -> p99 range per config, code, both concurrencies
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
for ax, c in zip(axes, CONCS):
    for i, cfg in enumerate(CONFIGS):
        r = R[(cfg, "code", c)]
        p50, p99 = r["p50_tpot_ms"], r["p99_tpot_ms"]
        ax.plot([p50, p99], [i, i], color=K_RAMP[cfg], lw=3, solid_capstyle="round")
        ax.plot(p50, i, "o", color=K_RAMP[cfg], ms=8, mec="white", mew=1)
        ax.plot(p99, i, "o", color=K_RAMP[cfg], ms=8, mec="white", mew=1, mfc="white")
        ax.annotate(f"{p50:.1f}", (p50, i), xytext=(-8, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=8, color=INK)
        ax.annotate(f"{p99:.1f}", (p99, i), xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(range(len(CONFIGS)), [LABEL[cfg] for cfg in CONFIGS])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 20 if c == 1 else 32)
    ax.set_xlabel("TPOT ms  (filled = p50, open = p99)")
    ax.set_title(f"code, concurrency {c}")
fig.suptitle("MTP widens the per-token tail: baseline p99 ≈ p50, MTP p99 is 1.3–2× p50",
             x=0.01, ha="left", fontweight="semibold", color=INK)
fig.tight_layout()
save(fig, "tpot_tail")

# 6. KV pool cost of the drafter
fig, ax = plt.subplots(figsize=(6.4, 3.6))
vals = [KV_TOKENS[cfg] / 1e6 for cfg in CONFIGS]
ax.bar(range(len(CONFIGS)), vals, color=[K_RAMP[c] for c in CONFIGS], width=0.6,
       edgecolor="white", linewidth=1)
for i, cfg in enumerate(CONFIGS):
    pct = 1 - KV_TOKENS[cfg] / KV_TOKENS["baseline"]
    txt = f"{KV_TOKENS[cfg]/1e6:.2f}M" + (f"\n−{pct:.1%}" if pct else "")
    ax.annotate(txt, (i, vals[i]), xytext=(0, 4), textcoords="offset points", ha="center",
                fontsize=8.5, color=INK)
ax.set_xticks(range(len(CONFIGS)), [LABEL[cfg] for cfg in CONFIGS])
ax.set_ylim(0, 2.4)
ax.grid(axis="x", visible=False)
ax.set_ylabel("GPU KV cache (M tokens @ 262k ctx, fp8)")
ax.set_title("The drafter's price: each extra draft step costs KV pool,\n15.5% at k=5 on a 96 GB card")
save(fig, "kv_pool_cost")

for (cfg, cell, c), r in sorted(R.items()):
    print(f"{cfg:9s} {cell:4s} c={c}  {r['output_throughput']:6.1f} tok/s  "
          f"x{speedup(cfg, cell, c):.2f}  tpot p50 {r['p50_tpot_ms']:.2f}  "
          f"acc.len {r.get('spec_decode_acceptance_length') or 0:.2f}")
