# M2 measurement: autoscaler, workload classes, scale-to-zero

This directory measures the autoscaler on Modal: two GPUs in one container, one trace, baselines, and
written pass rules. Modal bills the container for the whole run, so a saving here is GPU-seconds an
engine held, not money off the invoice. `runs.md.template` lists what a run may and may not claim.

| file | what it does |
|---|---|
| `trace.json` | the traces, as data, and the shortened static window |
| `loadgen.py` | open-loop client; one JSON line per request, `requests.meta.json` with its own lag |
| `engine.sh` | one engine for a slot, vLLM or sim, after the start pad |
| `probe.sh` | start times, the pad, the cancel and drain checks; writes `$M2/probe.env` |
| `derive_spec.py` | the `MANIFOLD_SPEC` for a start time |
| `cell.sh` | one cell, from no processes to `DONE` |
| `reclaim.sh` | an injected reclaim: stop file, SIGTERM, SIGKILL after the grace |
| `plan.sh` | probe and cells under a deadline; `--cost` is the budget check |
| `modal-run.sh` | starts `plan.sh` on Modal, detached, with a cap |
| `summarize.py` | metrics and rules P1 to P8 |
| `dryrun-local.sh` | everything above on the laptop, against sims |

## Before a run

1. Tests and the dry run, about 5 minutes, no GPU and no Hugging Face. The dry run ends with
   `dry run complete`. Its numbers say nothing about vLLM.

       uv run --no-sync pytest -q
       bench/m2/dryrun-local.sh

2. The Volume `manifold-m1` exists (verified 2026-11-07). Check that M1's state is on it. The plan needs
   the converted dataset, the pinned revisions and the bring-up facts, and the model weights under `hf/`.

       uv run --no-sync modal volume ls manifold-m1 m1
       uv run --no-sync modal volume ls manifold-m1 m1/data

   `m1/data/sharegpt.json`, `m1/revisions.env` and `m1/bringup-q35-9b.env` must be listed.

3. Budget. This exits 1 if the required steps exceed $15.63 or 95 minutes.

       bench/m2/plan.sh --cost

## Cost

$7.33/h for the container. The plan is 88 minutes and $10.75 through C6. C7 would make it 98 minutes,
so under a 95-minute cap the deadline drops it, which matches its priority. `plan.sh` logs each skip.
A rerun with the same RUN resumes, and finished cells cost nothing.

| step | min | answers |
|---|---|---|
| probe | 8 | S, pad, `idle_to_zero_s`, target, cancel and drain |
| C1 reconcile s0 | 20 | the claim |
| C2 static-2 s0 | 10 | the primary baseline, autoscaling alone |
| C3 static-1-blind s0 | 12 | what a never-firing `waiting` trigger gives |
| C4 static-2-blind s0 | 11 | the gate alone |
| C5 reclaim | 7 | reclaim, at 10 req/s with batch |
| C6 reconcile s1 | 20 | repeat |
| C7 static-2 s1 | 10 | repeat, first to drop |

## Run

Start from a clean tree if you can; `GIT_SHA` in each cell ends in `-dirty` otherwise. The image takes
the working tree as it is at launch.

    RUN=m2-$(date -u +%F)
    bench/m2/modal-run.sh "$RUN" 95

`modal-run.sh` repeats the budget check, fixes the deadline at CAP_MIN minus 5 in wall-clock time, and
returns once the app is detached. Use a new RUN for a new run: reusing one keeps its probe and its cells.

## Watch

    uv run --no-sync modal app list
    uv run --no-sync modal app logs <app-id>
    uv run --no-sync modal container exec <container-id> tail -f /workspace/m1/logs/m2-plan.log

In order:

- About 3 minutes in, the probe logs `slot 0 start`. Expect 128 to 234 s. A start under 60 s means the
  page cache was warm and the pad will be too small. Stop and use a fresh container.
- `cancel ok` and `drain ok`. If either fails, `plan.sh` stops before any cell, and the money spent is
  under $1. The probe sends SIGTERM to the engine leader, as the fleet does, so a failed drain would
  also fail in the cells. Fix it before running cells.
- `done: .../c1-reconcile-s0` about 28 minutes in. The first cold start ends at about t=45 + S, and
  the second replica cannot start before the first is READY plus `settle_s`.
- The final `summary.txt` is printed at the end of the log.

The idle phase is 180 s and `idle_to_zero_s` is S. If S is above about 175 s, the fleet never reaches
zero in that phase, and the trace has no second cold start. If S is near 170 s, the second start ends
after the last arrival, so those 360 requests are all held. Read the cold-window row with that in mind.

Through C6 the plan needs 88 minutes, and `modal-run.sh RUN 95` gives it 90, about $11.60 at most. If the
probe or C1 runs more than 2 minutes long, the deadline drops C6, the only repeat. Then report C1
without a repeat rather than buy a second container.

## Stop

    uv run --no-sync modal app stop <app-id>

Billing stops. The script commits the Volume only when the call ends, so a stopped call may lose cells
it finished. Cells without `DONE` are moved aside by the next run. A hang needs no action: the Function
timeout is CAP_MIN.

## Results

Copy the results from the Volume. `bench/m2/results/` is ignored by git.

    mkdir -p bench/m2/results
    uv run --no-sync modal volume get --force manifold-m1 m2/results bench/m2/
    uv run --no-sync modal volume get --force manifold-m1 m2/probe.env bench/m2/results/
    cd bench/m2 && ../../.venv/bin/python summarize.py "results/$RUN"

To run one finished cell again, remove its `DONE` on the Volume and resume the run with a short cap.
The probe is reused. Keep the old cell's local copy under another name first.

    uv run --no-sync modal volume rm manifold-m1 "m2/results/$RUN/c5-reclaim/DONE"
    bench/m2/modal-run.sh "$RUN" 15

Write the run up from `runs.md.template` as `runs-<date>.md`. The first one is `runs-2026-11-08.md`. A cell directory holds `requests.jsonl`,
`requests.meta.json`, `ticks.jsonl`, `decisions.jsonl`, `samples.tsv` (1 Hz `waiting` and `running`),
`events.jsonl`, `spec.json`, `env.json`, the router, autoscaler and engine logs.

## What the pad means

One container means one page cache. M1's logs show a first engine in a fresh container spending 128 to
234 s and a later start about 22 s. Measured as it is, a run would give every start after the first
the 22 s case and flatter the autoscaler. `probe.sh` measures both and writes
`START_PAD_S = S_FIRST - S_RESTART`, and `engine.sh` sleeps that long before the engine starts. So
every start the policy causes costs S, the fresh-container time. The pad is disclosed in the notes
and counts as GPU-seconds held, because a real replica would hold its GPU while it loaded.

Static and reclaim cells start their first engines without the pad, because those starts are set-up
and not measured. A reclaimed slot pays the pad.

`S` also sets `idle_to_zero_s` and the target through `derive_spec.py`, so the probe has to run in the
same container as the first cell. After a preemption, the new container's first start takes about S
plus the pad. If a resumed C1 fails P5 by about that much, rerun that cell and say so in the notes.

## Reading the rules

`summarize.py RUN_DIR` prints PASS, FAIL or NOT EVALUATED per rule with its numbers, C6 against C1, the
cold-window row and a harness row per cell. Check the harness rows before the rules: shed requests,
client errors and load generator lag.

`P2` starts when the backlog from the first cold start clears, because a cold start at S = 170 s costs
about 9% of the window on its own. The verdict prints the figure from t=165 beside it, and the cold-window
row and the `Backlog` row show what was left out. Publish those next to any `P2` result.

`P7` times the slot from the moment the engine exited, and prints the time from the reclaim as well,
because the drain alone can take 25 s.
