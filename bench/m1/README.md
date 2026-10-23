# bench/m1: the routing-policy comparison on a RunPod pod

This directory runs the M1 measurement from `docs/m1-router-spec.md`. Two vLLM 0.31.0 replicas
serve the same model, one per GPU on a single two-GPU pod. The manifold router sits in front of
them, and vLLM's own multi-turn benchmark replays ShareGPT conversations through the router. Each
cell fixes one policy, one load, one KV-cache pressure and one seed, and writes its results to
its own directory. This file is the runbook for pod day. Read it once end to end before you book
the pod.

## What it runs, and the budget

| Phase | Script | Wall hours | GPU-hours (2 GPUs) | USD at 2.09 | USD at 2.49 |
|---|---|---|---|---|---|
| Push and bootstrap | `push.sh --bootstrap` | 0.4 | 0.8 | 1.67 | 1.99 |
| Bring-up, Qwen3.5-9B | `bringup.sh q35-9b` | 0.5 | 1.0 | 2.09 | 2.49 |
| Calibration, 6 load levels | `calibrate.sh` | 0.7 | 1.4 | 2.93 | 3.49 |
| Main matrix, 48 cells | `matrix.sh` | 6.8 | 13.6 | 28.42 | 33.86 |
| Third seed on headline cells, 8 cells | `matrix.sh` | 1.15 | 2.3 | 4.81 | 5.73 |
| Controls W0 and W2, 8 cells | `controls.sh` | 1.15 | 2.3 | 4.81 | 5.73 |
| Dense control Qwen3-8B, bring-up and 8 cells | `control-model.sh` | 1.55 | 3.1 | 6.48 | 7.72 |
| Pull and tear down | `pull.sh` | 0.25 | 0.5 | 1.05 | 1.25 |
| Total | | 12.5 | 25.0 | 52.26 | 62.26 |

A cell takes about 8.5 minutes: 7 minutes of benchmark (1 minute of warm-up that the summary
drops, then 6 measured), plus the router restart, the cache reset and the canary. 2.09 USD per
GPU-hour is the rate M0 paid for one RTX PRO 6000 (`bench/m0-redux/runs-2026-09-24.md`). The
2.49 USD column is a higher rate for comparison, not a verified listing. Check the rate
when you book. The pod bills for both GPUs for the whole booking, including idle time between phases.

The spec budgeted 10.5 GPU-hours for two replicas sharing one GPU, plus a separate isolation
check on two devices. This plan puts each replica on its own GPU from the start, so the main
matrix is already the isolated configuration and that separate check is not needed.

## Prerequisites

1. `runpodctl` installed and configured. Run `runpodctl doctor` once in a real terminal; it
   reads the API key from stdin, so it fails under a non-interactive runner.
2. The RunPod SSH key at `~/.runpod/ssh/runpodctl-ssh-key`, or `RUNPOD_SSH_KEY` pointing at
   another key. `push.sh` and `pull.sh` read it.
3. A Hugging Face read token in `bench/m1/.env` (copy `.env.example`). Nothing is gated, but
   anonymous downloads of two 16 to 19 GB models are slow and rate limited.
4. The local checks pass:
   - `uv run bench/m1/summarize.py --self-test`
   - `bash bench/m1/tests/cell-stub.sh`
   - `bench/m1/fetch-vendor.sh` (needs network; checks the vendored benchmark against upstream)
   - `bench/m1/dryrun-local.sh` (bring-up, calibration and a mini matrix against two
     `manifold.sim` replicas; about 10 minutes on a laptop)

## Book the pod

```bash
runpodctl pod create --name m1-router \
  --gpu-id "NVIDIA RTX PRO 6000 Blackwell Server Edition" --gpu-count 2 \
  --image runpod/pytorch:1.4.1-cu1320-torch2130-ubuntu2404 \
  --volume-in-gb 150 --container-disk-in-gb 50 \
  --ports "22/tcp" --min-cuda-version 13.0 --wait
runpodctl ssh info <pod-id>
```

`--wait` blocks until SSH answers and then prints the pod. `runpodctl ssh info` prints the SSH
command; the host and port in it are the `HOST PORT` arguments below. Do not pass
`--docker-args`: RunPod appends it to the image ENTRYPOINT, and M0 lost a pod to that. The
volume holds the venv, both models (about 36 GB), ShareGPT and the results. Only port 22 is
exposed. The replicas, the router and the sims all bind to 127.0.0.1, because
`VLLM_SERVER_DEV_MODE=1` turns on `/reset_prefix_cache` and other dev endpoints, and the load
generator runs on the same pod.

When you are done, `runpodctl pod delete <pod-id>`. `pod stop` keeps billing for the disk.

## Push and bootstrap

```bash
bench/m1/push.sh HOST PORT --bootstrap
```

This copies the working tree (uncommitted edits included) to `/workspace/m1/repo`, records the
commit in `repo/GIT_SHA`, copies `.env`, and runs `bootstrap.sh`. Bootstrap is idempotent, so
rerun the same command after any failure. It installs uv and `vllm==0.31.0` with the cu130
torch build into `$VENV` (default `/workspace/.venv`; set `VENV=/root/.venv` in `bench/m1/.env`
to keep it on container disk, where vLLM starts much faster), installs the repo with its `hf` extra, downloads both
models and records each commit SHA in `/workspace/m1/revisions.env`, downloads and checks the
ShareGPT file, converts it, and writes `/workspace/m1/env-manifest.txt` and `pip-freeze.txt`.
Every later serve uses the recorded SHAs. To pin a different revision, set
`REV_Q35_9B=<sha>` or `REV_Q3_8B=<sha>` in the environment.

To reinstall exactly the packages of an earlier pod, put `FROZEN=1` in `bench/m1/.env`. Bootstrap
then installs from `bench/m1/results/manifest/pip-freeze.txt`, which `pull.sh` brought back and
`push.sh` sends along.

The ShareGPT conversion keeps conversations with at least two user turns, cuts them at ten
user turns, and drops any conversation with a message over 4,000 characters or an empty one
(`SHAREGPT_PREP_FLAGS` in `env.sh`), then samples 4,000 conversations. On the 2026-10-16
file, 24,179 conversations pass the filters, and 508 of the 4,000 sampled lose a trailing user
message. `prep_sharegpt.py` prints both counts. That keeps every replayed conversation near or below
20,000 tokens, under `MAX_MODEL_LEN` 32,768. The vendored converter and benchmark have four
known defects; `vendor/VENDOR.md` describes them, and `prep_sharegpt.py` and `run_bench.py`
work around them without changing the vendored files.

On the pod, every command below runs in an SSH session. The scripts source `env.sh`
themselves. For your own interactive commands, run `source /workspace/m1/repo/bench/m1/env.sh`
first.

## Bring-up

```bash
tmux new -d -s bringup 'bash -c "/workspace/m1/repo/bench/m1/bringup.sh q35-9b 2>&1 | tee -a /workspace/m1/logs/bringup.log"'
tail -f /workspace/m1/logs/bringup.log
```

`bringup.sh` starts replica A on GPU 0 (port 18001), waits for `/health`, then starts replica B
on GPU 1 (port 18002), both with vLLM sizing the KV pool. The first start compiles and captures
CUDA graphs; allow up to 20 minutes. It then records the startup facts, runs the canary on both
replicas, checks that a reply has no `<think>`, and measures router overhead with 200 single
requests direct to replica A and 200 through a router whose only backend is A. Output is in
`/workspace/m1/results/bringup/q35-9b/bringup-facts.txt`.

Go/no-go. Do not start calibration until every line holds:

- [ ] `bringup-facts.txt` has a `Setting attention block size to N tokens` line for both
      replicas, and `BLOCK_SOURCE=log`. Qwen3.5-9B is a hybrid model; vLLM 0.31.0 aligns its
      attention block to the Gated DeltaNet state size, so expect a value in the hundreds or
      thousands, not 16. Write the value down.
- [ ] `GPU KV cache size` and `Available KV cache memory` are recorded for both replicas and
      `BYTES_PER_TOKEN` is filled in `/workspace/m1/bringup-q35-9b.env`.
- [ ] The CUDA graph capture sizes match between replicas. `serve.sh` refuses to continue
      otherwise, and they must stay identical across every cell.
- [ ] Both canary lines say `canary ok`.
- [ ] The router overhead p50 delta is below 5 ms. A larger delta goes into every TTFT number,
      so find the cause in `overhead-router/router.log` before going on.
- [ ] The thinking check reply is a plain answer with no `<think>`. Read it.
- [ ] `bringup-facts.txt` says the router tokenizer has a chat template. `bringup.sh` stops if
      `AutoTokenizer` on the pinned snapshot loads without one.
- [ ] `nvidia-smi` shows both GPUs in use, and neither serve log has a traceback.

## Calibration and the knee

```bash
tmux new -d -s calib 'bash -c "/workspace/m1/repo/bench/m1/calibrate.sh calib 2>&1 | tee -a /workspace/m1/logs/calib.log"'
```

Calibration runs `least_load` on ShareGPT with vLLM's own pool size at
`CALIB_CLIENTS="2 4 8 16 32 64"` concurrent clients, one seed, 64 live conversations. Every
level divides 64, so every level runs exactly 64 live conversations. Then
`summarize.py --calibrate` prints, per level, achieved requests per second, p50 and p90 TTFT,
and the waiting queue's trend, and writes `/workspace/m1/knee.env`.

The knee rule: the knee is the lowest client count whose p90 TTFT exceeds the SLO
(`SLO_MS`, default 1,000 ms) or whose `vllm:num_requests_waiting` keeps growing over the
measured window. "Keeps growing" is a fitted slope above 0.5 requests per minute and a last
quarter at least 2 requests above the first. The matrix then runs at 30%, 70% and 95% of the
knee, rounded to whole clients, with request counts sized from the calibrated throughput to
fill `DURATION_S` (420 s).

Read the table before trusting `knee.env`. If no level fails, raise the top of the range and
rerun, for example `CALIB_CLIENTS="64 96 128" /workspace/m1/repo/bench/m1/calibrate.sh calib`; finished levels are
skipped. Above 64 clients each client holds one conversation, so the working set grows with
the client count; say so in the run record. If the rule picks a level that looks like noise, set it by hand with
`KNEE=<clients> /workspace/m1/repo/bench/m1/calibrate.sh calib`. Check the effective live conversation count in
the `active` column. The benchmark gives each client `ACTIVE / clients` conversations rounded
down, so `cell.sh` passes the multiple of the client count nearest to 64 and records both
numbers in `meta.json`. The matrix loads (30%, 70% and 95% of the knee) need not divide 64,
and `matrix.sh` warns when a load level runs more than 15% away from 64.

Then size the pressure axis:

```bash
uv run /workspace/m1/repo/bench/m1/size_kv.py /workspace/m1/results/calib/q35-9b/nocap
```

It prints the arithmetic: the mean block-rounded context of a live conversation over the last
half of each calibration cell, times the effective live conversation count, is the working set; "fits" is 2.0 times that and
"pressure" is 0.7 times that, per replica, converted to bytes with the bring-up bytes per
token. It writes `/workspace/m1/pressure-q35-9b.env`. Stop if it prints a warning. A pool below
`MAX_MODEL_LEN` makes vLLM refuse to start.

## Main matrix

```bash
RUN=m1-$(date -u +%F)
tmux new -d -s matrix "bash -c '/workspace/m1/repo/bench/m1/matrix.sh $RUN 2>&1 | tee -a /workspace/m1/logs/matrix.log'"
```

The matrix restarts the replicas once per pressure (`fits`, then `pressure`), then runs seeds 1
and 2, each at the three loads, with the four policies in an order rotated by seed. Each cell
lands in `/workspace/m1/results/$RUN/q35-9b/<pressure>/sharegpt/<policy>-c<clients>-s<seed>/`.
Each cell restarts the router with its policy, resets both prefix caches once the replicas
are idle, runs the canary, scrapes `/metrics`, samples the gauges every second, runs the
benchmark, scrapes `/metrics` again and writes `DONE` last.

tmux sessions: `matrix` is the driver, `vllm-a` and `vllm-b` are the replicas, `router` is the
router. The command lines tmux ran are in `/workspace/m1/run/*.sh`.

Watch progress:

```bash
tail -f /workspace/m1/logs/matrix.log
find /workspace/m1/results/$RUN -name DONE | wc -l
cat /workspace/m1/logs/failed-cells.txt
tmux attach -t matrix        # detach with C-b d
```

After a disconnect, the tmux sessions keep running; SSH in again and reattach. If the driver
itself died, run the same `matrix.sh $RUN` command again. Cells with `DONE` are skipped, and a
cell without `DONE` is moved aside to `<cell>.partial-<time>` and run again from the start.

When a cell fails, `matrix.sh` appends it to `failed-cells.txt` and goes on. Two failures in a
row stop the driver, since that usually means a replica died. For a failed cell, read
`bench.log`, `router.log` and the newest `/workspace/m1/logs/serve-*.log`. Fix the cause, then
rerun the same `matrix.sh $RUN` command, which retries only the cells without `DONE`. If a
replica is down, `matrix.sh` restarts both when it finds `/health` failing. A cell also fails
when no benchmark client reached its request quota, when a client process crashed, when the
benchmark returns fewer than 80% of its requests (`MIN_COMPLETE`), when the canary does not
hit, or when the benchmark runs past `BENCH_TIMEOUT_S`. The benchmark stops every client as
soon as the first one reaches its quota, so a clean cell returns a few percent fewer records
than it asked for.

Third seed on the headline cells (70% and 95% under pressure):

```bash
tmux new -d -s seed3 "bash -c 'SEEDS=3 LOADS=\"70 95\" PRESSURES=pressure /workspace/m1/repo/bench/m1/matrix.sh $RUN 2>&1 | tee -a /workspace/m1/logs/matrix.log'"
```

## Controls and the dense control

```bash
tmux new -d -s controls "bash -c '/workspace/m1/repo/bench/m1/controls.sh $RUN 2>&1 | tee -a /workspace/m1/logs/controls.log'"
tmux new -d -s control "bash -c '/workspace/m1/repo/bench/m1/control-model.sh $RUN 2>&1 | tee -a /workspace/m1/logs/control-model.log'"
```

`controls.sh` runs the four policies at 70% of the knee under pressure on W0 (ShareGPT first
turns only, `--max-turns 2`) and W2 (`w2-shared-prefix.json`: a 6,000-token prefix shared by
every conversation, 6 to 10 short user turns). W0 must show no `prefix_aware` gain beyond seed
noise; if it does, the harness is wrong and the matrix is rerun. W2 is where `prefix_aware`
should gain most.

`control-model.sh` brings up Qwen3-8B (dense, full attention on every layer, vLLM's default
block size), sizes its pressure pool from the Qwen3.5-9B calibration working set with its own
block size and bytes per token, and runs the 70% and 95% pressure cells for the four policies
with the primary model's client counts. The dense control matters more than its size suggests. The
router hashes whole blocks only, so on Qwen3.5-9B any prompt shorter than one block (estimated
at about 524 tokens before alignment in the spec's D4; bring-up reads the real value) gets a prefix score of 0 on both replicas and `prefix_aware` routes it
exactly as `least_load` does. Early ShareGPT turns are often that short, so the gain should appear from the third or
fourth turn of a conversation on. On Qwen3-8B the block
is 16 tokens and every turn carries prefix information. Its bring-up has the same gates as
above, except that
no `Setting attention block size` line is expected: `BLOCK_SOURCE=default` means vLLM's default
of 16, which you confirm from the serve log.

## Pull results and tear down

```bash
bench/m1/pull.sh HOST PORT
uv run bench/m1/summarize.py bench/m1/results/<RUN>
runpodctl pod delete <pod-id>
```

`pull.sh` copies `/workspace/m1/results/` into `bench/m1/results/`, the logs into
`bench/m1/logs/`, and the manifests and generated `.env` files (never the secrets file) into
`bench/m1/results/manifest/`. `summarize.py` writes `summary.csv` next to the results and
prints the headline tables. Then write `bench/m1/runs-<date>.md` from `runs.md.template`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `nvcc: command not found` over SSH | Non-login shells lack `/usr/local/cuda/bin`. `env.sh` adds it; source it in interactive shells. |
| vLLM fails to load `libcudart.so.13` or other cu13 libraries | The venv's `nvidia/cu13/lib` must be on `LD_LIBRARY_PATH`. `env.sh` sets it, and the tmux launch scripts in `/workspace/m1/run/` source `env.sh`. |
| `Stale file handle (os error 116)` during `uv pip install` | The volume is network backed. `bootstrap.sh` retries 5 times and prints the filesystem type. Rerun `push.sh HOST PORT --bootstrap`, or book with a pod-local volume. `UV_LINK_MODE=copy` is already set. |
| `/health` takes many minutes | The first start compiles and captures CUDA graphs. `serve.sh` waits 20 minutes and prints the log tail on timeout. Later starts reuse the compile cache. |
| The pod never exposes SSH | `--docker-args` was passed and appended to the ENTRYPOINT. Book again without it, from a `runpod/pytorch` image. |
| `runpodctl doctor` hangs or fails | It reads the API key from stdin. Run it in a real terminal. |
| Downloads go to the wrong place | The RunPod image presets `HF_HOME`; `env.sh` sets `/workspace/hf` explicitly. |
| `POST /reset_prefix_cache` returns 404 | `VLLM_SERVER_DEV_MODE=1` did not reach the server. Check `/workspace/m1/run/vllm-a.sh`. |
| Canary fails on Qwen3.5-9B | The prompt must span whole engine blocks. Check that `CUR_BLOCK_TOKENS` in `/workspace/m1/current.env` is the logged block size; the canary sizes its prompt from it. |
| vLLM refuses to start under `pressure` | The pool is smaller than one `MAX_MODEL_LEN` request. `size_kv.py` warns about this; lower `MAX_MODEL_LEN` for every phase, or raise `ACTIVE`. |
| The benchmark hangs with all clients idle | A client process died; the vendored benchmark then waits forever. `BENCH_TIMEOUT_S` kills it and fails the cell. Read `bench.log` for the traceback. |
| Watching the pod from Claude Code | The sandbox blocks SSH and Monitor cannot bypass it. Use a background Bash command with the sandbox bypass that exits on a condition, and end remote polls with `; true` so a failed `test -f` does not look like an SSH failure. |

## Before any number goes in a post

Answer each for the number, in `runs-<date>.md`, next to the number:

1. What limits it? Name the bottleneck (queueing, prefill compute, KV capacity, decode) with
   the evidence: queue-time and prefill-time histogram means, waiting samples, KV usage.
2. Were both sides tuned? Every policy runs on the same replicas, flags, capture sizes and
   router build. Say so, or say what differed.
3. Did it error? Report the error count and the decision-log status for every cell in the
   comparison, and the bench record count against the expected count.
4. Does it reproduce? Report the spread across seeds, and compare only cells from one pod.
5. Did the work happen? A TTFT gain needs a higher prefix hit rate and a lower prefill-time
   mean to be explained. One without either is not explained and is not published as a win.
   Check the per-turn backend coverage and the same-replica share in `summary.csv`.

## Files

- `env.sh` holds paths, ports, serve flags, model tags and run knobs; `lib.sh` holds the shared shell helpers.
- `push.sh`, `pull.sh` and `fetch-vendor.sh` run locally. Everything else runs on the pod.
- `bootstrap.sh`, `serve.sh`, `stop.sh` and `router.sh` set up and run the servers.
- `cell.sh` runs one cell. `bringup.sh`, `calibrate.sh`, `matrix.sh`, `controls.sh` and `control-model.sh` are the drivers.
- `size_kv.py` sizes the pressure axis. `summarize.py` builds `summary.csv`, the headline tables and `knee.env`.
- `run_bench.py`, `prep_sharegpt.py` and `sampler.py` wrap the vendored benchmark and converter and sample the gauges.
- `w2-shared-prefix.json` is the W2 workload. `vendor/` is the unmodified vLLM benchmark and `VENDOR.md`.
- `tests/` holds the summary fixture, the `cell.sh` stub test and the dry-run vLLM shim. `dryrun-local.sh` is the laptop rehearsal.
