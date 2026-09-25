# M0-redux — Qwen3.8-27B-NVFP4 baseline vs MTP on one RTX PRO 6000 (2026-09-21)

Purpose: restart M0 with a clean one-variable comparison. M0 and M0-part2 (`bench/m0/runs.md`,
`bench/m0-part2/runs.md`) were dominated by RTX 5090 toolchain and memory-fit problems, and
part2's MTP result (28 tok/s, 37.6% acceptance) came from a server squeezed into 32 GB with
`--enforce-eager`. Here the recipe command runs unmodified on a 96 GB card, the load generator is
`vllm bench serve` with real-text datasets, and the only variable is `--speculative-config`.

**Follow-up (2026-09-24):** repeats, sampling, thinking on, TRITON_ATTN, more categories, long
input and the TTFT explanation are in `runs-2026-09-24.md`. Two findings there revise this file:
the chat TTFT gap is a CUDA-graph capture-size effect, and absolute numbers differ between pods.

Scope actually run: code (InstructCoder) and chat (mt-bench), greedy, thinking off.
Not run: math, summarize/translate/RAG, long-context buckets, sampling T=0.7, thinking on,
repeats. `bench.sh` defines most of those cells already.

## Environment

- Pod: RunPod `xlrsclye40uuxm`, 1x NVIDIA RTX PRO 6000 Blackwell Server Edition 97,887 MiB,
  $2.09/hr, 32 vCPU / 125 GB, host CPU AMD EPYC 9554, kernel 6.8.0-124, Ubuntu 24.04.4
- Image: `runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404-cluster` — nvcc 13.0.88, driver 580.159.04
- Stack: vLLM **0.29.0**, torch 2.13.0+cu130, transformers 5.17.0, flashinfer-python 0.6.18,
  triton 3.7.1, numpy 2.3.5, datasets 5.0.1. Full list: `results/pip-freeze.txt`;
  manifest: `results/env-manifest.txt`
- Model: `nvidia/Qwen3.8-27B-NVFP4` @ `482ca0f3832238542f8f5295dde86b5f22711d80`.
  Weights on GPU 20.82 GiB. The checkpoint has one MTP layer (`mtp_num_hidden_layers: 1`),
  excluded from quantization; with k>1 the same layer is run k times per draft.
- Datasets: `likaixin/InstructCoder` @ `6a778a720284d6520b56bd03d5c3070930d41071`,
  `philschmid/mt-bench` @ `bd19d60b9afe5201563a1008f58959322187d924`
- Kernel proof lines (every start): `Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM`;
  `Using FLASHINFER attention backend`; FlashInfer `decode_backend=xqa,
  kv_cache_dtype=torch.float8_e4m3fn, arch=sm120`. No Marlin fallback.

Install, as run on the pod:

```bash
cd /workspace && uv venv .venv --python 3.12
uv pip install -U vllm --torch-backend auto
uv pip install -U "transformers>=5.8.0"
uv pip install "vllm[bench]==0.29.0" --torch-backend auto   # datasets etc. for `vllm bench serve`
```

The `bench` extra left vLLM/torch/transformers unchanged and moved numpy 2.5.3 → 2.3.5 and
fsspec 2026.9.0 → 2026.6.0. All timed runs used the post-install stack in `pip-freeze.txt`.

## Config

Server: the recipe command verbatim
(<https://recipes.vllm.ai/Qwen/Qwen3.8-27B?variant=nvfp4_nvidia>) plus `--seed 0 --port 8000`.
See `serve.sh`; environment in `env.sh` (`VLLM_USE_RUST_FRONTEND=1`, CUDA bin on PATH,
venv `nvidia/cu13/lib` on `LD_LIBRARY_PATH`).

```bash
vllm serve nvidia/Qwen3.8-27B-NVFP4 --kv-cache-dtype fp8 --max-model-len 262144 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 --enable-chunked-prefill --async-scheduling \
  --enable-prefix-caching --tensor-parallel-size 1 --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml --reasoning-parser qwen3 --mm-encoder-tp-mode data \
  --seed 0 --port 8000
# MTP configs add only:
  --speculative-config '{"method":"mtp","num_speculative_tokens":K}'   # K = 1,2,3,5,7
```

`--gpu-memory-utilization` left at the 0.92 default.

Client: `bench.sh CELL CONCURRENCY`, which runs

```bash
vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --skip-chat-template \
  --model nvidia/Qwen3.8-27B-NVFP4 --port 8000 <dataset flags> --temperature 0 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --num-prompts N --max-concurrency C --num-warmups 4 --seed S \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --save-result ...
# code: --dataset-name hf --dataset-path likaixin/InstructCoder --hf-output-len 512
# chat: --dataset-name hf --dataset-path philschmid/mt-bench   --hf-output-len 512
```

Protocol:

- One server start per config; closed-loop load at fixed concurrency 1 and 8 (8 = `--max-num-seqs`).
- Code: c=1 uses seed 1 / 80 prompts, c=8 uses seed 8 / 160 prompts. Different seeds give
  different prompts within a server (no prefix-cache hits between cells); the same seeds are
  used for every config, so input tokens are identical across configs.
- Chat: mt-bench has 80 prompts, so c=1 and c=8 both use seed 1 / 80 prompts and share prompts
  within a server. **c=8 chat TTFT includes prefix-cache hits and is not comparable.**
- No `--ignore-eos`: requests stop at EOS. Thinking off verified on a manual request
  (`reasoning: null`).
- 4 warmup requests per cell, discarded by the tool. **No repeats**: each number is one run.
- Drivers: `run-code.sh` (19:54–20:27Z), `run-k7-chat.sh` (20:34–21:08Z). 0 failed requests,
  0 failed cells.

## Results

Speedup = output tok/s relative to baseline at the same cell and concurrency. Acceptance
length = mean tokens emitted per decode step (1 + accepted draft tokens per draft).

### Code — InstructCoder (input 12,064 tok @ c=1, 26,016 @ c=8; mean output ≈ 173–186 tok)

| Config   | Conc. | Output tok/s | Speedup | TPOT p50 ms | TPOT p99 ms | TTFT p50 ms | Accept. len | Per-position acceptance % |
| -------- | ----- | ------------ | ------- | ----------- | ----------- | ----------- | ----------- | ------------------------- |
| baseline | 1     | 59.9         | 1.00x   | 15.77       | 16.03       | 188         | –           | –                         |
| mtp k=1  | 1     | 96.0         | 1.60x   | 9.34        | 9.66        | 191         | 1.98        | 98                        |
| mtp k=2  | 1     | 127.9        | 2.14x   | 6.73        | 7.35        | 167         | 2.91        | 97/93                     |
| mtp k=3  | 1     | 149.8        | 2.50x   | 5.53        | 6.38        | 194         | 3.79        | 97/94/88                  |
| mtp k=5  | 1     | 185.5        | 3.10x   | 4.14        | 5.51        | 192         | 5.21        | 96/91/84/79/72            |
| mtp k=7  | 1     | 201.2        | 3.36x   | 3.67        | 5.96        | 201         | 6.33        | 95/90/83/76/69/63/56      |
| baseline | 8     | 332.9        | 1.00x   | 22.12       | 27.14       | 200         | –           | –                         |
| mtp k=1  | 8     | 478.3        | 1.44x   | 15.18       | 22.14       | 209         | 1.97        | 97                        |
| mtp k=2  | 8     | 579.0        | 1.74x   | 12.26       | 19.38       | 210         | 2.89        | 97/92                     |
| mtp k=3  | 8     | 630.2        | 1.89x   | 11.05       | 15.01       | 220         | 3.76        | 97/92/87                  |
| mtp k=5  | 8     | 719.9        | 2.16x   | 9.48        | 12.74       | 218         | 5.21        | 96/90/84/78/72            |
| mtp k=7  | 8     | 731.4        | 2.20x   | 8.84        | 17.92       | 236         | 6.28        | 95/88/82/75/69/63/56      |

### Chat — mt-bench (input 6,452 tok; mean output ≈ 360–370 tok)

| Config   | Conc. | Output tok/s | Speedup | TPOT p50 ms | TPOT p99 ms | TTFT p50 ms | Accept. len | Per-position acceptance % |
| -------- | ----- | ------------ | ------- | ----------- | ----------- | ----------- | ----------- | ------------------------- |
| baseline | 1     | 61.7         | 1.00x   | 15.79       | 15.81       | 170         | –           | –                         |
| mtp k=3  | 1     | 135.6        | 2.20x   | 6.70        | 10.39       | 109         | 3.00        | 84/66/50                  |
| mtp k=5  | 1     | 151.0        | 2.45x   | 5.68        | 9.50        | 110         | 3.57        | 83/63/47/36/28            |
| mtp k=7  | 1     | 144.0        | 2.33x   | 5.79        | 10.67       | 114         | 3.84        | 81/61/45/34/26/20/16      |
| baseline | 8     | 377.9        | 1.00x   | 19.42       | 21.06       | (192)       | –           | –                         |
| mtp k=3  | 8     | 729.3        | 1.93x   | 9.06        | 16.66       | (212)       | 3.01        | 84/66/51                  |
| mtp k=5  | 8     | 764.0        | 2.02x   | 8.44        | 15.13       | (198)       | 3.56        | 82/63/47/36/28            |
| mtp k=7  | 8     | 723.5        | 1.91x   | 8.81        | 15.94       | (150)       | 3.80        | 81/60/44/33/26/20/16      |

Parenthesised TTFT: prefix-cache contaminated, see Protocol.

### KV pool and startup per config (262k max-model-len, fp8 KV, util 0.92)

| Config   | GPU KV cache tokens   | Max concurrency @262k |
| -------- | --------------------- | --------------------- |
| baseline | 2,018,810 / 2,012,783 | 7.70x / 7.68x         |
| mtp k=1  | 1,839,501             | 7.02x                 |
| mtp k=2  | 1,808,197             | 6.90x                 |
| mtp k=3  | 1,777,892 / 1,773,499 | 6.78x / 6.77x         |
| mtp k=5  | 1,705,360             | 6.51x                 |
| mtp k=7  | 1,648,563             | 6.29x                 |

MTP k=5 costs 15.5% of the KV pool. At these workloads (≤ 8 seqs, < 1k tokens each) the pool
is never a constraint. Engine init: 487 s on the first start, 140–170 s afterwards.

## Findings

1. MTP is a throughput and latency gain on this setup at every k tested: 1.4x–3.4x output tok/s.
   This reverses part2's result, which was measured under eager mode and a KV pool MTP had halved.
2. The gain depends on the text. At k=5, code-edit output accepts 5.21 tokens per step, chat 3.57.
   InstructCoder is near a best case: edited code copies much of its input.
3. Best k: 5 for chat (k=7 is slower at both concurrencies). For code, k=7 adds 8% at c=1 and
   1.6% at c=8 over k=5, with worse p99 TPOT at c=8 (12.7 → 17.9 ms). The recipe's k=5 is a sound default.
4. The speedup shrinks as the batch fills: code k=5 3.10x at c=1, 2.16x at c=8.
5. Acceptance length is independent of concurrency (5.21 vs 5.21; 3.57 vs 3.56), as expected.
6. MTP widens the TPOT distribution. Baseline p99 ≈ p50; with MTP p99 is 1.5–2x p50, because
   tokens per step vary with acceptance.
7. Per-position acceptance decays with depth (chat: 83 → 28% over 5 positions), consistent with
   one MTP layer being reused for every position.

## Caveats and open items

- Single run per cell, no variance estimate. Differences of a few percent (code k=5 vs k=7 at
  c=8) are within what a repeat could reverse.
- (Explained 2026-09-24: CUDA-graph capture sizes, see `runs-2026-09-24.md` finding 8.)
- Unexplained: chat c=1 TTFT p50 is 170 ms on baseline and ~110 ms on all MTP configs. Not prefix
  caching (first chat cell on each server). Code TTFT shows no such gap. Needs a repeat.
- Output lengths differ by up to ~2% between configs (no `--ignore-eos`), so compare tok/s and
  TPOT, not request rate. ITL is not comparable under spec decode (several tokens per chunk).
- Startup log: `Fused multi-step draft decode is not supported by attention backend(s)
  FLASHINFER; falling back to rebuilding attention metadata between draft steps.` The MTP numbers
  are on that fallback path; `TRITON_ATTN` is untested.
- Rust frontend ignores `enable_auto_tool_choice` and `structured_outputs_config` (warnings at
  start). No effect on these cells.
- Closed-loop fixed concurrency, not M0's Poisson open-loop loadgen, so no knee/saturation curve
  and no direct comparison with M0 tables.
- Greedy only. Sampling at the model card's non-thinking settings (T=0.7, top_p=0.8, top_k=20)
  will lower acceptance.
- A 16-prompt mt-bench smoke run on the first k=5 server (before the `bench` extra changed numpy)
  was deleted and is not in `results/`.

## Files

- `bootstrap.sh` — pod setup: uv, pinned stack, model + datasets, manifest. `FROZEN=1` installs `results/pip-freeze.txt` exactly
- `push.sh`, `pull.sh` — run locally: copy scripts (and `.env`) to the pod; copy results and logs back
- `env.sh` — paths, `MODEL_REVISION`, loads `/workspace/m0-redux/.env` if present
- `.env.example` — template for `HF_TOKEN`. The real `.env` is gitignored
- `serve.sh`, `stop.sh`, `bench.sh`, `run-code.sh`, `run-k7-chat.sh` — everything run on the pod
- `results/<config>/<cell>-c<N>-greedy-thinkoff.{json,log}` — `vllm bench serve` result JSON + console output
- `results/env-manifest.txt`, `results/pip-freeze.txt`
- `charts.py` — `uv run bench/m0-redux/charts.py` renders `charts/*.{svg,png}` from the result JSONs: speedup_vs_k, throughput_by_k, acceptance_vs_speedup, acceptance_by_position, tpot_tail, kv_pool_cost
- `logs/serve-*.log` — full server log per start; `logs/run-*.log` — driver logs
- 2026-09-24 additions: `run-deferred.sh`, `run-triton.sh`, `run-categories.sh`, `run-long.sh`,
  `run-sweep.sh` (drivers); `probe-ttft.py`, `ttft-test.py` (TTFT); results in `results/2026-09-24/`

## Reproduce

```bash
# local: rent 1x RTX PRO 6000 with the image above, 80 GB volume at /workspace, then
cp .env.example .env && $EDITOR .env            # optional: HF_TOKEN for higher download limits
./push.sh <pod-ip> <ssh-port> --bootstrap       # ~5 min on a fresh pod
ssh ... "tmux new -d -s matrix '/workspace/m0-redux/scripts/run-code.sh 2>&1 | tee /workspace/m0-redux/logs/run-code.log'"
# wait for /workspace/m0-redux/DONE (~35 min), then run-k7-chat.sh the same way (~35 min)
./pull.sh <pod-ip> <ssh-port>
runpodctl pod delete <pod-id>
```

HF token, either way: `HF_TOKEN=...` in `.env` (copied by `push.sh`, mode 600), or on the pod
`HF_HOME=/workspace/hf hf auth login`. The 2026-09-21 run used the second. Nothing here is
gated; anonymous downloads work, with lower rate limits.

Differences between these scripts and the recorded run, none of which change the server config:
`bootstrap.sh` was written afterwards from the manual install (it pins `--torch-backend=cu130`
and `transformers==5.17.0`, where the run used `auto` and `>=5.8.0`, resolving to the same
versions); `serve.sh` now passes `--revision` (the run resolved `main` to that same commit).
Checked on the same pod after the run: bootstrap is a no-op on an installed pod, and a baseline
start with `--revision` gave 60.2 tok/s on an 8-prompt cell against 59.9 recorded. The
fresh-pod path and `FROZEN=1` have not been exercised.
