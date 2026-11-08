"""Print the MANIFOLD_SPEC JSON for a measured start time S.

target_interactive is the in-flight count per replica at knee_rps - ramp * (S + up_window + margin),
read off the M1 calibration table and rounded down, so the second replica is ready before the ramp
reaches the knee. idle_to_zero_s is S, the rent-or-buy break-even.

    derive_spec.py --s 170 [--floor 0] [--max-replicas 2] [--override '{"tick_s": 0.25}']
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from itertools import pairwise

from manifold.autoscaler.policy import Spec

# (in flight per replica, req/s per replica), calib/q35-9b/nocap with clients divided by 2.
KNEE_TABLE = ((0, 0.0), (16, 3.5), (32, 5.9), (48, 7.7), (64, 9.0), (96, 10.35), (128, 10.6))
KNEE_RPS = 10.35
RAMP_RPS_PER_S = 10 / 360
MARGIN_S = 10.0


def in_flight_at(rps: float) -> float:
    for (n0, r0), (n1, r1) in pairwise(KNEE_TABLE):
        if rps <= r1:
            return n0 + (n1 - n0) * (rps - r0) / (r1 - r0)
    return KNEE_TABLE[-1][0]


def target_interactive(s: float, ramp: float = RAMP_RPS_PER_S, knee_rps: float = KNEE_RPS,
                       up_window_s: float = Spec.up_window_s,
                       margin_s: float = MARGIN_S) -> int:
    rps = knee_rps - ramp * (s + up_window_s + margin_s)
    if rps <= 0:
        raise SystemExit(f"derive_spec: the ramp reaches the knee before a replica can start "
                         f"(S={s}, ramp={ramp}/s)")
    return math.floor(in_flight_at(rps))


def derive(s: float, floor: int, max_replicas: int, ramp: float, override: dict) -> Spec:
    spec = replace(Spec(), floor=floor, max_replicas=max_replicas, idle_to_zero_s=s,
                   target_interactive=target_interactive(s, ramp))
    return replace(spec, **override)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--s", type=float, required=True, help="seconds one start takes, pad included")
    p.add_argument("--floor", type=int, default=0)
    p.add_argument("--max-replicas", type=int, default=2)
    p.add_argument("--ramp-rps-per-s", type=float, default=RAMP_RPS_PER_S)
    p.add_argument("--override", default="{}", help="JSON object of Spec fields, applied last")
    a = p.parse_args()
    spec = derive(a.s, a.floor, a.max_replicas, a.ramp_rps_per_s, json.loads(a.override))
    print(json.dumps(asdict(spec)))


if __name__ == "__main__":
    main()
