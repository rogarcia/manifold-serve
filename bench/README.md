# bench

Benchmark scripts and run records. Each milestone publishes its scripts, a `runs*.md` record with
the exact commands, and its charts. Raw output (CSVs, JSON, logs) stays local.

Protocol per run:

1. Record hardware (GPU model, host, region), model, vLLM version, replica config.
2. Warm up: 60s at the target rate, results discarded.
3. Measure: 120s or more per policy, identical seed and trace parameters across policies.
4. Write a `runs.md` entry with the full command lines and the resulting numbers.

M1 compares round_robin, least_load, session_hash and prefix_aware across load and KV-cache
pressure. Its runbook is `bench/m1/README.md`.
