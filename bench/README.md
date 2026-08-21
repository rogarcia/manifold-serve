# bench

Reproducible benchmark runs. Every number in the top-level README comes from a CSV in this
directory plus the exact command that produced it.

Protocol per run:

1. Record hardware (GPU model, host, region), model, vLLM version, replica config.
2. Warm up: 60s at the target rate, results discarded.
3. Measure: 120s or more per policy, identical seed and trace parameters across policies.
4. Commit the CSV plus a `runs.md` entry with the full command lines.

Comparisons that ship with M1: round_robin vs least_outstanding vs prefix_aware at low,
medium, and saturating arrival rates.
