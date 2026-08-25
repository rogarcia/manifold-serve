# M0-part2 — native NVFP4 path on one RTX 5090 (2026-08-24)

Purpose: rerun M0's benchmark with the native Blackwell sm_120 FP4 stack that M0's cu128 image
couldn't build. Same model, GPU, loadgen, protocol. M0 baseline: `bench/m0/runs.md`.

## Environment
- Pod: RunPod secure, RTX 5090 32,607 MiB, $0.99/hr, 32 vCPU / 124 GB host
- Image: `runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2404` — nvcc **12.9.86**, driver 580.126.09 (CUDA 13.0)
- Stack: vLLM **0.26.0** (0.27.x PyPI wheels are CUDA-13-linked, unusable on this toolkit),
  torch 2.13.0+cu129, transformers 5.15.1, flashinfer 0.6.16.post3 (autotune cache 0.6.14), ninja 1.13
- Kernel proof lines: `Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM`;
  FlashInfer decode `arch=sm120`, `kv_cache_dtype=torch.float8_e4m3fn`; fp4_gemm autotuned 64 configs
- Weights on GPU: 24.7 GiB (incl. vision tower; M0: 23.94)

## Config
Recipe UI command ("Verified on RTX 5090": TP1 + parsers + `--mm-encoder-tp-mode data` + MTP-3,
max-model-len auto) **does not start** on one 5090: profile OOM at 262k ctx with CUDA graphs
(135 MiB free, 800 MiB KV alloc). The page's 1x5090 guide variant is the real config:
`--enforce-eager --max-model-len 32768 --kv-cache-dtype fp8`, plus `--gpu-memory-utilization 0.95`
(0.92 leaves 1.65 GiB KV < 1.87 GiB minimum once the MTP head + vision tower are resident).

## Results (loadgen: Poisson open-loop, 32 sessions, 120 s/rate, thinking off)

### A/B at a glance
| | M0 marlin+TRITON | part2 native no-MTP | part2 native +MTP-3 |
|---|---|---|---|
| saturated output tok/s | 115–123 | **176 (+43%)** | 28 |
| knee (TTFT detaches) | ~1.8 req/s | 2–4 req/s (rate 2: TTFT p50 287 ms) | <1 req/s |
| TPOT p50 @ rate 0.5 | 63 ms | 86.5 ms | 99.3 ms |
| single-stream TPOT p50 | 84 ms (anomaly) | 87.4 ms | 98.8 ms |
| KV pool @32k / max concurrency | 2.12 GiB / 53,399 tok | 3.64 GiB / **92,235 / 2.81x** | 2.59 GiB / 44,600 / 1.36x |
| prefix cache hits (sweep) | 0 | **43,904 tok** (~1% of 4.35M queried) | 0 |
| nvidia-smi at saturation | n/a | bursts 98% SM / 518 W | starved 32% / 220 W |

### Per-rate, native no-MTP (full CSVs in this dir)
| rate | ok/total | TTFT p50/p99 ms | TPOT p50 ms | out tok/s |
|---|---|---|---|---|
| 0.5 | 57/59 | 252 / 304 | 86.5 | 28.7 |
| 1 | 122/122 | 269 / 342 | 88.1 | 64.7 |
| 2 | 237/238 | 287 / 831 | 97.0 | 119.5 |
| 4 | 452/453 | 22,730 / 39,431 | 96.8 | 171.1 |
| 8 | 892/892 | 99,558 / 207,433 | 97.6 | 176.2 |

MTP run per-rate is in `mtp/`: throughput flat ~28 tok/s from rate 1, TTFT p50 74 s at rate 2,
443/888 errors at rate 8 (queue overflow on a 1.36x-concurrency pool).

### MTP diagnosis (counters, sweep traffic)
Acceptance 50,355/134,049 = **37.6%** (recipe reported 0.75+ on its workload); per-position
0.61 / 0.34 / 0.18. ~2.1 tokens/step didn't pay for 3 uncompiled draft passes per step in eager
mode, and the draft head's KV layers halved the pool. Smoke test (predictable text) showed 100%
acceptance — MTP acceptance is workload-dependent; measure on your own traffic.

## Findings
1. **Native kernels shift the win from latency to throughput.** Batch-1 decode is *slower* than
   Marlin (87 vs 63–84 ms TPOT — Marlin's dequant path is excellent when memory-bound), but the
   native fp8-KV FlashInfer stack + 1.7x KV pool sustains ~3x concurrency: knee 1.8 -> >2 req/s,
   saturation 123 -> 176 tok/s (+43%).
2. **Prefix caching finally hits** (43.9k tokens vs M0's zero) — session histories cross the
   1,568-token hybrid-GDN block boundary and get reused with the larger pool.
3. **MTP-3 is a net loss here**: 37.6% acceptance, throughput collapse via KV starvation.
   Candidate rescue (untested): MTP with CUDA graphs on the drafter, or num_speculative_tokens=1-2.
4. Recipe UI vs guide text disagree for 1x5090; the guide (eager, 32k) is correct. Reported upstream? TODO.

## Toolchain gotchas (part2's ladder, for the post)
1. `--torch-backend=auto` matches the *driver* (13.0) not the *toolkit* (nvcc 12.9) -> cu132 torch. Pin cu129.
2. vLLM >=0.27 PyPI wheels link libcudart.so.13; no cu129 variant. Pin vllm==0.26.*.
3. vLLM 0.26 ships its own CUDA-13 runtime via `nvidia-cuda-runtime` pip dep; venvs need
   `LD_LIBRARY_PATH=<venv>/site-packages/nvidia/cu13/lib` (three CUDA versions coexist by design).
4. FlashInfer JIT shells out to `ninja` (not a vllm dep) — install ninja+cmake or die mid-profile.
5. RunPod `--terminate-after` had no effect (pod ran past 4h). Manual `pod delete` is the only teardown.

## Cost
Session actual: **$5.21** (balance $16.995 → $11.789; ~4.5 h GPU @ $0.99 + volume/overhead). M0: $3.00.
Two-experiment total: $8.21. currentSpendPerHr 0 confirmed after delete.
