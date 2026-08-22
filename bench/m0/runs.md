# M0 runs — Qwen3.8-27B-NVFP4, 1x RTX 5090 (RunPod)

| run | date | vLLM | rate (req/s) | sessions | TTFT p50 | TTFT p99 | out tok/s | csv |
|-----|------|------|--------------|----------|----------|----------|-----------|-----|
| | | | | | | | | |

Environment: vLLM 0.27.1 · driver 580.126.09 · RTX 5090 32,607 MiB · runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 · pod fgdrx1xpi6o9sc (US, 16 vCPU/110 GB) · $0.99/hr secure cloud

Protocol: warm-up 60 s discarded; 120 s per rate; sweep 0.5,1,2,4,8 req/s; seed 42.
- Single-stream (rate 0.2, 1 session, after the sweep): TTFT p50 307 ms, TPOT p50 84.3 ms (15.5 tok/s).
  Slower than the 0.5 req/s run (63 ms) — anomaly; suspects: the 85 client-timed-out requests from the
  8 req/s stage still draining server-side, or clocks after 361 W sustained. Re-measure on a fresh
  server next session before quoting a single-stream number.
- Charts: `uv run bench/m0/charts.py` → bench/m0/charts/{ttft_vs_rate,tpot_throughput,ttft_by_turn}.{svg,png}
