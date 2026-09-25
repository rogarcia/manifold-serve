# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Render the charts for the 2026-09-24 follow-up.  Run:  uv run bench/m0-redux/charts-2026-09-24.py

Reads `results/2026-09-24/<config>/<cell>-c<N>-<sampling>-think<on|off>-s<seed>.json` and
`results/2026-09-24/ttft-test.jsonl`, plus the 09-21 anchor cells in `results/<config>/`, and
draws the findings in runs-2026-09-24.md. Same style and palette as charts.py.

Outputs bench/m0-redux/charts/2026-09-24/*.svg and *.png.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).parent
RES = HERE / "results" / "2026-09-24"
OLD = HERE / "results"
OUT = HERE / "charts" / "2026-09-24"
OUT.mkdir(parents=True, exist_ok=True)

# Same reference palette as charts.py (validated: blue/orange pass all six checks, light mode).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, BASE = "#1c2026", "#5c6673", "#e3e6e2", "#8a929c"

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


def load(cfg: str, name: str, root: Path = RES) -> dict:
    with open(root / cfg / f"{name}.json") as f:
        return json.load(f)


def thr(cfg: str, name: str) -> float:
    return load(cfg, name)["output_throughput"]


def speedup(cfg: str, name: str, base: str = "baseline") -> float:
    return thr(cfg, name) / thr(base, name)


def acc(cfg: str, name: str) -> float:
    return load(cfg, name)["spec_decode_acceptance_length"]


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.svg")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", OUT / name)


def g(cell: str, c: int = 1, seed: int = 1, sampling: str = "greedy", think: str = "off") -> str:
    return f"{cell}-c{c}-{sampling}-think{think}-s{seed}"


# 1. Acceptance and speedup by category (k=5, c=1). Two measures -> two panels, shared rows.
CATS = [("code", "code edits (InstructCoder)"), ("math", "math (GSM8K)"),
        ("translate", "translation"), ("rag", "RAG"), ("chat", "chat (mt-bench)"),
        ("summarize", "summarization"), ("qa", "QA")]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
ys = list(range(len(CATS)))
for ax, metric in zip(axes, ("acc", "speedup")):
    for y, (cell, _) in zip(ys, CATS):
        v = acc("mtp-k5", g(cell)) if metric == "acc" else speedup("mtp-k5", g(cell))
        ax.barh(y, v, height=0.6, color=BLUE, edgecolor="white", linewidth=1)
        ax.annotate(f"{v:.2f}" + ("×" if metric == "speedup" else ""), (v, y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(ys, [lbl for _, lbl in CATS])
axes[0].invert_yaxis()
axes[0].set_xlim(0, 6)
axes[0].set_xlabel("acceptance length (tokens per decode step, max 6)")
axes[0].set_title("acceptance length")
axes[1].set_xlim(0, 3.9)
axes[1].axvline(1, color=BASE, lw=1)
axes[1].set_xlabel("output tok/s relative to baseline")
axes[1].set_title("speedup")
fig.suptitle("MTP k=5 by content: the more predictable the output, the larger the gain\n"
             "(c=1, greedy; translation outputs are ~26 tokens, so TTFT limits its speedup)",
             x=0.01, ha="left", fontweight="semibold", color=INK)
fig.tight_layout()
save(fig, "category_acceptance_speedup")

# 2. Long input: acceptance flat, decode speedup flat, end-to-end speedup falls with prefill.
BUCKETS = ["1k", "8k", "16k", "32k"]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
xs = list(range(len(BUCKETS)))
e2e = [speedup("mtp-k5", g(f"long{b}")) for b in BUCKETS]
dec = [load("baseline", g(f"long{b}"))["median_tpot_ms"] / load("mtp-k5", g(f"long{b}"))["median_tpot_ms"]
       for b in BUCKETS]
ax = axes[0]
for vals, color, lbl in ((dec, BLUE, "decode (TPOT p50 ratio)"), (e2e, ORANGE, "end-to-end (output tok/s)")):
    ax.plot(xs, vals, "-o", color=color, lw=2, ms=7, mec="white", mew=1, label=lbl)
    ax.annotate(f"{vals[-1]:.2f}×", (xs[-1], vals[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=INK)
ax.set_xticks(xs, BUCKETS)
ax.set_xlim(-0.3, 3.6)
ax.set_ylim(1, 3.4)
ax.set_xlabel("SPEED-Bench input bucket (mixed category)")
ax.set_ylabel("MTP k=5 speedup over baseline")
ax.set_title("speedup")
ax.legend(frameon=False, loc="lower left", fontsize=9)
ax = axes[1]
al = [acc("mtp-k5", g(f"long{b}")) for b in BUCKETS]
ttft = [load("baseline", g(f"long{b}"))["median_ttft_ms"] / 1000 for b in BUCKETS]
ax.plot(xs, al, "-o", color=BLUE, lw=2, ms=7, mec="white", mew=1)
for x, v in zip(xs, al):
    ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 8), textcoords="offset points", ha="center",
                fontsize=8.5, color=INK)
ax.set_xticks(xs, [f"{b}\nTTFT {t:.1f} s" for b, t in zip(BUCKETS, ttft)])
ax.set_ylim(0, 6)
ax.set_xlabel("input bucket (baseline TTFT p50)")
ax.set_ylabel("acceptance length")
ax.set_title("acceptance length")
fig.suptitle("Longer input does not lower acceptance up to 32k; end-to-end speedup falls\n"
             "because prefill, which MTP does not speed up, grows (c=1, 16 prompts per bucket)",
             x=0.01, ha="left", fontweight="semibold", color=INK)
fig.tight_layout()
save(fig, "long_input")

# 3. TTFT and CUDA-graph capture sizes (unique-prefix prompts, max_tokens=1, FlashInfer).
tt = {}
with open(RES / "ttft-test.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if r["max_tokens"] == 1:
            tt[(r["config"], r["set"])] = (statistics.median(r["client_ttft_ms"]),
                                           statistics.mean(r["prompt_tokens"]))
fig, ax = plt.subplots(figsize=(7, 3.9))
SETS = [("chat", "chat prompts"), ("code", "code prompts")]
CFGS = [("baseline", "baseline\ngraphs up to 16 tokens", BASE),
        ("mtp-k5", "MTP k=5\ngraphs up to 96 tokens", BLUE)]
w = 0.36
for i, (s, _) in enumerate(SETS):
    for j, (cfg, _, color) in enumerate(CFGS):
        v, _ = tt[(cfg, s)]
        x = i + (j - 0.5) * (w + 0.04)
        ax.bar(x, v, width=w, color=color, edgecolor="white", linewidth=1)
        ax.annotate(f"{v:.0f} ms", (x, v), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK)
ax.set_xticks([0, 1], [f"{lbl}\n(~{tt[('baseline', s)][1]:.0f} prompt tokens)" for s, lbl in SETS])
ax.set_ylim(0, 135)
ax.grid(axis="x", visible=False)
ax.set_ylabel("client TTFT p50 (ms), max_tokens=1")
ax.legend(handles=[Line2D([], [], color=c, lw=6, label=l) for _, l, c in CFGS], frameon=False,
          loc="upper right", fontsize=9, ncols=2)
ax.set_title("MTP's lower TTFT is a CUDA-graph effect: with k=5 vLLM captures graphs up to 96\n"
             "tokens (16 on baseline), so a ~85-token prefill runs as one graph; ~180 does not")
save(fig, "ttft_cudagraph")

# 4. TRITON_ATTN vs FlashInfer: gain per cell, baseline and MTP k=5.
CELLS = [(g("chat"), "chat c=1"), (g("code"), "code c=1"), (g("code", 8, 8), "code c=8")]
fig, ax = plt.subplots(figsize=(7, 3.8))
for i, (name, _) in enumerate(CELLS):
    for j, (cfg, color) in enumerate((("baseline", BASE), ("mtp-k5", BLUE))):
        v = 100 * (thr(f"{cfg}-triton", name) / thr(cfg, name) - 1)
        x = i + (j - 0.5) * (w + 0.04)
        ax.bar(x, v, width=w, color=color, edgecolor="white", linewidth=1)
        ax.annotate(f"+{v:.1f}%", (x, v), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=8.5, color=INK)
ax.set_xticks(range(len(CELLS)), [lbl for _, lbl in CELLS])
ax.set_ylim(0, 20)
ax.grid(axis="x", visible=False)
ax.set_ylabel("output tok/s gain, TRITON_ATTN over FlashInfer (%)")
ax.legend(handles=[Line2D([], [], color=BASE, lw=6, label="baseline"),
                   Line2D([], [], color=BLUE, lw=6, label="MTP k=5")],
          frameon=False, loc="upper left", fontsize=9)
ax.set_title("TRITON_ATTN is faster on this card; MTP gains most at c=8, where FlashInfer\n"
             "lacks fused multi-step draft decode (single run per Triton cell)")
save(fig, "triton_vs_flashinfer")

# 5. What lowers the speedup: sampling and thinking (k=5, dumbbell from reference to variant).
PAIRS = [("code c=1: greedy → T=0.7", g("code"), g("code", sampling="qwen")),
         ("code c=8: greedy → T=0.7", g("code", 8, 8), g("code", 8, 8, sampling="qwen")),
         ("chat c=1: greedy → T=0.7", g("chat"), g("chat", sampling="qwen")),
         ("GSM8K c=1: thinking off → on", g("math"), g("math", think="on"))]
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for y, (lbl, a, b) in enumerate(PAIRS):
    va, vb = speedup("mtp-k5", a), speedup("mtp-k5", b)
    ax.plot([va, vb], [y, y], color=BLUE, lw=2.5, solid_capstyle="round")
    ax.plot(va, y, "o", color=BLUE, ms=8, mec="white", mew=1)
    ax.plot(vb, y, "o", color=BLUE, ms=8, mec=BLUE, mew=1.5, mfc="white")
    ax.annotate(f"{va:.2f}×", (va, y), xytext=(8, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=INK)
    ax.annotate(f"{vb:.2f}×", (vb, y), xytext=(-8, 0), textcoords="offset points", va="center",
                ha="right", fontsize=8.5, color=MUTED)
ax.set_yticks(range(len(PAIRS)), [p[0] for p in PAIRS])
ax.invert_yaxis()
ax.grid(axis="y", visible=False)
ax.set_xlim(2, 3.5)
ax.set_xlabel("MTP k=5 speedup over baseline  (filled = reference, open = variant)")
ax.set_title("Sampling at T=0.7 costs 3–5% of the speedup; thinking on costs 18%")
save(fig, "sampling_thinking")

# 6. Seed repeats: k=5 and k=7 speedup per seed, code c=1 and c=8.
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
for ax, (c, seeds) in zip(axes, ((1, (1, 2, 3)), (8, (8, 9, 10)))):
    for cfg, color, off in (("mtp-k5", BLUE, -0.1), ("mtp-k7", ORANGE, 0.1)):
        vals = [speedup(cfg, g("code", c, s)) for s in seeds]
        ax.plot([i + off for i in range(3)], vals, "o", color=color, ms=8, mec="white", mew=1)
        ax.annotate(f"{statistics.mean(vals):.2f}× mean", (2 + off, vals[-1]), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(3), [f"seed {s}" for s in seeds])
    ax.set_xlim(-0.4, 3.1)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"code, concurrency {c}")
    ax.set_ylabel("speedup over baseline")
axes[0].set_ylim(3.1, 3.6)
axes[1].set_ylim(2.3, 2.65)
axes[0].legend(handles=[Line2D([], [], color=BLUE, marker="o", lw=0, ms=8, label="k=5"),
                        Line2D([], [], color=ORANGE, marker="o", lw=0, ms=8, label="k=7")],
               frameon=False, loc="upper left", fontsize=9)
fig.suptitle("k=7 beats k=5 on code for every seed, at both concurrencies",
             x=0.01, ha="left", fontweight="semibold", color=INK)
fig.tight_layout()
save(fig, "seed_repeats")

# 7. Two pods, same stack and prompts: acceptance reproduces, speedup does not (k=5).
ANCH = [("code-c1-greedy-thinkoff", g("code"), "code c=1"),
        ("code-c8-greedy-thinkoff", g("code", 8, 8), "code c=8"),
        ("chat-c1-greedy-thinkoff", g("chat"), "chat c=1")]
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
for ax, metric in zip(axes, ("speedup", "acc")):
    for i, (old, new, _) in enumerate(ANCH):
        if metric == "speedup":
            vo = load("mtp-k5", old, OLD)["output_throughput"] / load("baseline", old, OLD)["output_throughput"]
            vn = speedup("mtp-k5", new)
        else:
            vo = load("mtp-k5", old, OLD)["spec_decode_acceptance_length"]
            vn = acc("mtp-k5", new)
        for j, (v, color) in enumerate(((vo, BASE), (vn, BLUE))):
            x = i + (j - 0.5) * (w + 0.04)
            ax.bar(x, v, width=w, color=color, edgecolor="white", linewidth=1)
            ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(ANCH)), [a[2] for a in ANCH])
    ax.grid(axis="x", visible=False)
axes[0].set_ylim(0, 3.8)
axes[0].set_title("MTP k=5 speedup (×)")
axes[1].set_ylim(0, 6.2)
axes[1].set_title("acceptance length")
axes[0].legend(handles=[Line2D([], [], color=BASE, lw=6, label="pod 1 (2026-09-21, driver 580)"),
                        Line2D([], [], color=BLUE, lw=6, label="pod 2 (2026-09-24, driver 595)")],
               frameon=False, loc="upper right", fontsize=8.5)
fig.suptitle("Same stack, same prompts, two pods: acceptance reproduces (±0.04); the c=8 speedup\n"
             "does not (2.16× vs 2.40×), so compare speedups within one pod",
             x=0.01, ha="left", fontweight="semibold", color=INK)
fig.tight_layout()
save(fig, "pod_comparison")
