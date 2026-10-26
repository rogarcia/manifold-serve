# bench/m1 on Modal

This directory runs the M1 measurement on Modal instead of a RunPod pod. The bench scripts are
the same ones `bench/m1/README.md` describes, run unchanged, on the same hardware: two NVIDIA
RTX PRO 6000 in one container, one vLLM replica per GPU. Read that runbook first. This file
only covers what differs on Modal. The RunPod runbook and scripts stay as they are; nothing here
changes them.

## How it maps onto the pod runbook

| Pod | Modal |
|---|---|
| `runpodctl pod create`, SSH | `modal run`, one container per call |
| `push.sh`: tar of the working tree | `app.py` mounts the working tree at `/root/repo` at container start, uncommitted edits included; `GIT_SHA` is written from the local `git` |
| `/workspace`: the pod's network volume | `/workspace`: the Modal Volume `manifold-m1`, kept between calls |
| `bootstrap.sh` installs the venv | the image carries `/root/.venv`, built from `bench/m1/results/manifest/pip-freeze.txt` (the 2026-10-23 pod); `bootstrap.sh` still downloads models and data and writes the manifest |
| `bench/m1/.env` copied to the pod | `app.py` reads `HF_TOKEN` from `bench/m1/.env` and passes it as a Modal Secret; nothing else from that file is used |
| `pull.sh HOST PORT` | `bench/m1/modal/pull.sh`, into `bench/m1/modal/results/` and `bench/m1/modal/logs/` |
| `runpodctl pod delete` | nothing to delete: billing stops when the call returns |

Every call starts a fresh container, so the replicas, router and tmux sessions of one call are
gone by the next. Chain the steps that need running replicas in one `--cmd`, as below. State
that later steps read (`revisions.env`, `bringup-*.env`, `knee.env`, `pressure-*.env`,
results) lives in `/workspace/m1` on the Volume, as it does on the pod.

## Cost

Modal's list prices on 2026-10-23: RTX PRO 6000 $3.03 per GPU-hour, $0.047 per physical
core-hour, $0.008 per GiB-hour. `app.py` asks for 2 GPUs, 16 cores and 64 GiB, which is
$6.06 + $0.75 + $0.51 = about $7.33 per hour while a call runs.

| Phase | Wall hours (pod runbook) | USD on Modal |
|---|---|---|
| Bootstrap (model and data download) | 0.4 | 2.9 |
| Bring-up, Qwen3.5-9B | 0.5 | 3.7 |
| Calibration | 0.7 | 5.1 |
| Main matrix, 48 cells | 6.8 | 49.8 |
| Third seed, controls, dense control | 3.85 | 28.2 |
| Total | 12.25 | about 90 |

The Starter plan's $30 monthly credit covers about 4 hours: bootstrap, both bring-ups and
calibration fit with room to spare. The Volume holds about 40 GB (both models, ShareGPT,
results) at $0.09 per GiB-month; delete it after the run (see Clean up).

Modal can preempt any GPU call; there is no non-preemptible option for GPUs. A preempted call
restarts from the start of its `--cmd` in a new container. `cell.sh` skips cells with `DONE` and
moves a partial cell aside, and `calibrate.sh` and `matrix.sh` skip finished cells, so a
restart repeats at most one cell plus the replica start. Bring-up reruns from the start.

## Prerequisites

1. `uv sync --all-extras` (the Modal client is the `modal` extra) and `uv run modal setup`
   once. `uv run modal token info` should show your workspace.
2. A payment method on the Modal account; GPU calls need one even when credits cover them.
3. `HF_TOKEN` in `bench/m1/.env`, as for the pod.
4. The local checks from the pod runbook pass.

## Run

All commands run from the repo root. `--detach` keeps the call running if the laptop sleeps
or the terminal closes; logs still stream while the command is attached. `--cmd` runs with
bash in `bench/m1`, and its output also goes to `/workspace/m1/logs/modal-<time>-<script>.log`.

The first call builds the image (about 10 GB of wheels), which takes several minutes once.
Later calls reuse it until `pip-freeze.txt` or `app.py`'s image steps change.

Optional free check of the image, scripts and Volume: the laptop dry run on a CPU container.

```bash
uv run modal run bench/m1/modal/app.py --cpu --cmd "./dryrun-local.sh"
```

Bootstrap: downloads both models, ShareGPT and pg1184 onto the Volume, records the revisions
and writes the manifest. Watch for `HF token: set`.

```bash
uv run modal run --detach bench/m1/modal/app.py --cmd "./bootstrap.sh"
```

Bring-up for Qwen3.5-9B, then the go/no-go list in the pod runbook. The first start compiles
and captures CUDA graphs; the compile cache is kept on the Volume (`VLLM_CACHE_ROOT`).

```bash
uv run modal run --detach bench/m1/modal/app.py --cmd "./bringup.sh q35-9b"
uv run modal volume get manifold-m1 m1/results/bringup/q35-9b/bringup-facts.txt -
```

Calibration starts its own replicas when none are running, which is always true in a new
container. Read the table, then size the pressure axis in a CPU call:

```bash
uv run modal run --detach bench/m1/modal/app.py --cmd "./calibrate.sh calib"
uv run modal run bench/m1/modal/app.py --cpu --cmd "uv run ./size_kv.py /workspace/m1/results/calib/q35-9b/nocap"
```

To set the knee or the matrix's live conversations by hand without starting GPUs (calibrate.sh
would start the replicas first), call `summarize.py` directly in a CPU call. The 2026-10-23
run used this, with the knee at the throughput plateau (see the amendment in
`docs/m1-router-spec.md`):

```bash
uv run modal run bench/m1/modal/app.py --cpu --cmd 'uv run ./summarize.py --calibrate /workspace/m1/results/calib/q35-9b/nocap --knee 192 --active 384 --write-knee /workspace/m1/knee.env && source ./env.sh && uv run ./size_kv.py /workspace/m1/results/calib/q35-9b/nocap'
```

`bench/m1/modal/profile.sh [CLIENTS] [POLICY]` runs one cell while sampling GPU use and CPU per
process, to check what limits throughput and how close the router is to a full core.
`bench/m1/modal/overhead.sh LABEL [ORDER]` repeats the bring-up's router overhead step.

The matrix, controls and dense control, as in the pod runbook. Set `RUN` once and keep it for
every step of the run, including steps on a later day:

```bash
RUN=m1-$(date -u +%F)
uv run modal run --detach bench/m1/modal/app.py --cmd "./matrix.sh $RUN"
uv run modal run --detach bench/m1/modal/app.py --cmd "SEEDS=3 LOADS='70 95' PRESSURES=pressure ./matrix.sh $RUN"
uv run modal run --detach bench/m1/modal/app.py --cmd "./controls.sh $RUN"
uv run modal run --detach bench/m1/modal/app.py --cmd "./control-model.sh $RUN"
```

For a run capped in money rather than cells, `bench/m1/modal/budget.sh RUN CAP_MIN` starts
`bench/m1/budget.sh` (the priority-ordered cell list) detached, with the Function timeout set to
`CAP_MIN` and the budget.sh deadline fixed at launch time minus 5 minutes. A preempted call
restarts against the same deadline, skips finished cells, and exits once nothing more fits.

```bash
bench/m1/modal/budget.sh m1-$(date -u +%F) 115
```

Steps can be chained in one call to avoid a second replica start, for example
`--cmd "./bringup.sh q35-9b && ./calibrate.sh calib"`, but then nobody reads the go/no-go list
before calibration spends GPU time.

## Watch a running call

```bash
uv run modal app list                                  # the manifold-m1 app and its id
uv run modal app logs <app-id>                         # reattach to the log stream
uv run modal container list                            # container id of the running call
uv run modal container exec <container-id> tail -f /workspace/m1/logs/matrix.log
uv run modal shell <container-id>                      # a shell in the running container
uv run modal app stop <app-id>                         # stop the call; billing stops
```

`modal container exec` and `modal shell <container-id>` work only while the call runs. Results
on the Volume can be read at any time with `modal volume ls manifold-m1 m1/results` and
`modal volume get`, or in a CPU container: `uv run modal shell bench/m1/modal/app.py::bench_cpu`.

## Pull and clean up

```bash
bench/m1/modal/pull.sh
```

This copies `m1/results` and `m1/logs` from the Volume into `bench/m1/modal/` and the manifests
into `bench/m1/modal/results/manifest/`. It never writes into `bench/m1/results/`, which holds
the RunPod runs.

When the results are pulled and checked, delete the Volume to stop the storage charge:

```bash
uv run modal volume delete manifold-m1
```
