# M0 runs — Qwen3.8-27B-NVFP4, 1x RTX 5090 (RunPod)

Environment: vLLM 0.27.1 · driver 580.126.09 · RTX 5090 32,607 MiB · runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 · US, 16 vCPU/110 GB · $0.99/hr secure cloud

Protocol: warm-up 60 s discarded; 120 s per rate; sweep 0.5,1,2,4,8 req/s; seed 42.
- Single-stream (rate 0.2, 1 session, after the sweep): TTFT p50 307 ms, TPOT p50 84.3 ms (15.5 tok/s).
  Slower than the 0.5 req/s run (63 ms) — anomaly; suspects: the 85 client-timed-out requests from the
  8 req/s stage still draining server-side, or clocks after 361 W sustained. Not re-measured, so no
  single-stream number is quoted.
- Charts: bench/m0/charts/{ttft_vs_rate,tpot_throughput,ttft_by_turn}.{svg,png}
