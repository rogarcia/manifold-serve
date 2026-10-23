# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Size the KV pool for the fits and pressure conditions from bring-up facts and calibration.

  uv run size_kv.py CALIB_ROOT [--model-tag q35-9b] [--m1 /workspace/m1] [--active 64]

Working set = live conversations (the calibration cells' effective -k) x mean context of one live conversation, where a conversation's context
is a request's input plus output tokens, rounded up to whole engine blocks, over the last half
of each calibration cell. fits = 2.0 x working set, pressure = 0.7 x working set, per replica.
Bytes per token come from the uncapped pool line at bring-up (available KV memory / KV tokens).
Writes $M1/pressure-<tag>.env and prints the arithmetic.
"""

import argparse
import json
import math
import os
import statistics
from pathlib import Path

FITS, PRESSURE = 2.0, 0.7


def read_env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("'\"")
    return out


def cell_contexts(cell: Path, block: int) -> list[int]:
    recs = json.loads((cell / "stats.json").read_text())
    recs.sort(key=lambda r: r["start_time_ms"])
    late = recs[len(recs) // 2 :]
    return [math.ceil((r["input_num_tokens"] + r["output_num_tokens"]) / block) * block for r in late]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calib_root", type=Path, help="calibration results tree (cells with stats.json and DONE)")
    ap.add_argument("--model-tag", default=os.environ.get("MODEL_TAG", "q35-9b"))
    ap.add_argument("--m1", type=Path, default=Path(os.environ.get("M1", "/workspace/m1")))
    ap.add_argument("--active", type=int, help="live conversations in the matrix; default: $ACTIVE when set "
                    "(env.sh, raised by knee.env), else the most the calibration cells ran with")
    ap.add_argument("--max-model-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "32768")))
    a = ap.parse_args()

    facts = read_env(a.m1 / f"bringup-{a.model_tag}.env")
    block, pool = int(facts["BLOCK_TOKENS"]), int(facts["KV_TOKENS"])
    bpt = float(facts["BYTES_PER_TOKEN"]) if facts.get("BYTES_PER_TOKEN") else None

    cells = sorted(p.parent for p in a.calib_root.rglob("DONE") if (p.parent / "stats.json").exists())
    if not cells:
        raise SystemExit(f"no finished cells under {a.calib_root}")
    actives = sorted({json.loads((c / "meta.json").read_text())["active_effective"] for c in cells})
    if a.active is None and os.environ.get("ACTIVE"):
        a.active = int(os.environ["ACTIVE"])
    if a.active is None:
        a.active = max(actives)
        if len(actives) > 1:
            print(f"WARNING: calibration cells ran with different live conversation counts {actives}; using {a.active}")
    pooled: list[int] = []
    print(f"bring-up ({a.model_tag}): block {block} tokens, uncapped pool {pool:,} tokens per replica, "
          f"{'%.1f bytes per token' % bpt if bpt else 'no byte size (sim)'}")
    for c in cells:
        ctx = cell_contexts(c, block)
        pooled += ctx
        print(f"  {c.relative_to(a.calib_root)}: {len(ctx)} late requests, mean context {statistics.mean(ctx):,.0f} tokens")
    mean_ctx = statistics.mean(pooled)
    ws = a.active * mean_ctx
    print(f"mean context (block-rounded) {mean_ctx:,.0f} tokens x {a.active} live conversations = working set {ws:,.0f} tokens")

    env = {"WORKING_SET_TOKENS": round(ws)}
    for label, mult in (("fits", FITS), ("pressure", PRESSURE)):
        tokens = round(mult * ws)
        env[f"KV_TOKENS_{label}"] = tokens
        line = f"{label}: {mult} x {ws:,.0f} = {tokens:,} tokens"
        if bpt:
            env[f"KV_BYTES_{label}"] = round(tokens * bpt)
            line += f" x {bpt:.1f} B = {env[f'KV_BYTES_{label}']:,} bytes ({env[f'KV_BYTES_{label}'] / 2**30:.2f} GiB)"
        print(line)
        if tokens > pool:
            print(f"  WARNING: {label} asks for more than the uncapped pool ({pool:,} tokens); vLLM will not start")
        if tokens < a.max_model_len:
            print(f"  WARNING: {label} is below MAX_MODEL_LEN {a.max_model_len:,}; vLLM refuses a pool that cannot hold one full-length request")
    out = a.m1 / f"pressure-{a.model_tag}.env"
    out.write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
