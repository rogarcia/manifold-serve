# M0 runs — Qwen3.8-27B-NVFP4, 1x RTX 5090 (RunPod)

| run | date | vLLM | rate (req/s) | sessions | TTFT p50 | TTFT p99 | out tok/s | csv |
|-----|------|------|--------------|----------|----------|----------|-----------|-----|
| | | | | | | | | |

Environment: vLLM 0.27.1 · driver 580.126.09 · RTX 5090 32,607 MiB · runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 · pod fgdrx1xpi6o9sc (US, 16 vCPU/110 GB) · $0.99/hr secure cloud

Protocol: warm-up 60 s discarded; 120 s per rate; sweep 0.5,1,2,4,8 req/s; seed 42.
