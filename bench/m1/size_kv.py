# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Size the KV pool for the fits and pressure conditions from bring-up facts and calibration.

  uv run size_kv.py CALIB_ROOT [--model-tag q35-9b] [--m1 /workspace/m1] [--active 64]

The unit is the engine block, because that is what vLLM allocates. One live conversation holds
its context (a request's input plus output tokens, rounded up to whole blocks, over the last half
of each calibration cell) plus STATE_BLOCKS: on a hybrid model, one recurrent-state block in each
DeltaNet group. Working set = live conversations (the calibration cells' effective -k) x mean
blocks of one conversation. fits = 2.0 x working set, pressure = 0.7 x working set, per replica.
Bytes per block are BLOCK_BYTES, or BYTES_PER_TOKEN from bring-up x the block size on a dense
model. STATE_BLOCKS and BLOCK_BYTES come from env.sh.
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


def cell_blocks(cell: Path, block: int) -> list[int]:
    """Context blocks of each request in the last half of the cell, recurrent state excluded."""
    recs = json.loads((cell / "stats.json").read_text())
    recs.sort(key=lambda r: r["start_time_ms"])
    late = recs[len(recs) // 2 :]
    return [math.ceil((r["input_num_tokens"] + r["output_num_tokens"]) / block) for r in late]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calib_root", type=Path, help="calibration results tree (cells with stats.json and DONE)")
    ap.add_argument("--model-tag", default=os.environ.get("MODEL_TAG", "q35-9b"))
    ap.add_argument("--m1", type=Path, default=Path(os.environ.get("M1", "/workspace/m1")))
    ap.add_argument("--active", type=int, help="live conversations in the matrix; default: $ACTIVE when set "
                    "(env.sh: the caller's, else MATRIX_ACTIVE from knee.env), else the most the calibration cells ran with")
    ap.add_argument("--max-model-len", type=int, default=int(os.environ.get("MAX_MODEL_LEN", "32768")))
    ap.add_argument("--state-blocks", type=int, default=os.environ.get("STATE_BLOCKS"),
                    help="recurrent-state blocks per conversation; default: $STATE_BLOCKS from env.sh")
    ap.add_argument("--block-bytes", type=int, default=int(os.environ.get("BLOCK_BYTES") or 0) or None)
    a = ap.parse_args()
    if a.state_blocks is None:
        raise SystemExit("STATE_BLOCKS is unset: source env.sh first, or pass --state-blocks")
    a.state_blocks = int(a.state_blocks)

    facts = read_env(a.m1 / f"bringup-{a.model_tag}.env")
    block, pool = int(facts["BLOCK_TOKENS"]), int(facts["KV_TOKENS"])
    bpt = float(facts["BYTES_PER_TOKEN"]) if facts.get("BYTES_PER_TOKEN") else None
    block_bytes = a.block_bytes or (round(bpt * block) if bpt else None)
    uncapped = int(float(facts["KV_GIB"]) * 2**30 / block_bytes) if block_bytes and facts.get("KV_GIB") else pool // block

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
    print(f"bring-up ({a.model_tag}): block {block} tokens, {a.state_blocks} state blocks per conversation, "
          f"uncapped pool {uncapped:,} blocks per replica, "
          f"{f'{block_bytes:,} bytes per block' if block_bytes else 'no byte size (sim)'}")
    for c in cells:
        blocks = cell_blocks(c, block)
        pooled += blocks
        print(f"  {c.relative_to(a.calib_root)}: {len(blocks)} late requests, mean context {statistics.mean(blocks):.2f} blocks")
    per_conv = statistics.mean(pooled) + a.state_blocks
    ws = a.active * per_conv
    print(f"{statistics.mean(pooled):.2f} context + {a.state_blocks} state = {per_conv:.2f} blocks per conversation"
          f" x {a.active} live conversations = working set {ws:,.0f} blocks")

    env = {"WORKING_SET_BLOCKS": round(ws)}
    for label, mult in (("fits", FITS), ("pressure", PRESSURE)):
        blocks = round(mult * ws)
        env[f"KV_BLOCKS_{label}"] = blocks
        # Attention capacity in tokens: the sim's pool and the canary's prompt cap read it.
        env[f"KV_TOKENS_{label}"] = blocks * block
        line = f"{label}: {mult} x {ws:,.0f} = {blocks:,} blocks"
        if block_bytes:
            env[f"KV_BYTES_{label}"] = blocks * block_bytes
            line += f" x {block_bytes:,} B = {env[f'KV_BYTES_{label}']:,} bytes ({env[f'KV_BYTES_{label}'] / 2**30:.2f} GiB)"
        print(line)
        if blocks > uncapped:
            print(f"  WARNING: {label} asks for more than the uncapped pool ({uncapped:,} blocks); vLLM will not start")
        if blocks * block < a.max_model_len:
            print(f"  WARNING: {label} is below MAX_MODEL_LEN {a.max_model_len:,}; vLLM refuses a pool that cannot hold one full-length request")
    out = a.m1 / f"pressure-{a.model_tag}.env"
    out.write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
