"""Run the bench/m1 scripts on Modal: two RTX PRO 6000 in one container, /workspace on a Volume.

The same shell scripts as the RunPod runbook run unchanged. This file replaces the parts that
assume an SSH pod: the image carries the venv (built from the pod's pip-freeze.txt), the
repo is mounted at container start instead of pushed with tar, and /workspace is a Volume, so
models, data, state files and results survive between calls. See README.md next to this file.

    uv run modal run --detach bench/m1/modal/app.py --cmd "./bootstrap.sh"
    uv run modal run bench/m1/modal/app.py --cpu --cmd "./dryrun-local.sh"
"""

import datetime
import os
import shlex
import subprocess
from pathlib import Path

import modal

BENCH = Path(__file__).resolve().parent.parent
REPO = BENCH.parent.parent
VOLUME = os.environ.get("M1_VOLUME", "manifold-m1")
GPU = "RTX-PRO-6000:2"
# CPU is physical cores (two vCPUs each). The bench runs one process per client (up to 64)
# next to two vLLM engines and the router, so leave headroom rather than measure contention.
# M1_CPU overrides it for one call.
CPU = int(os.environ.get("M1_CPU", "16"))
MEMORY_MIB = 64 * 1024
# A hard cap on one container's run time, so a hang cannot spend past the budget. M1_TIMEOUT_S
# overrides it for one call (modal/budget.sh sets it).
TIMEOUT_S = int(os.environ.get("M1_TIMEOUT_S", 24 * 3600))

# Same CUDA toolkit major and minor as the RunPod image of 2026-10-23 (nvcc 13.2): FlashInfer
# JIT-compiles its kernels with it. Modal's driver supports CUDA 13.3.
CUDA_IMAGE = "nvidia/cuda:13.2.1-devel-ubuntu24.04"
FREEZE = Path(__file__).parent / "pip-freeze.txt"

# Directories never sent to the container: VCS, environments, caches, local results and logs.
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".scratch",
             ".claude", ".firecrawl", "results", "logs", "node_modules"}


def _skip(p: Path) -> bool:
    return bool(SKIP_DIRS.intersection(p.parts)) or p.name == ".env"


image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install("tmux", "curl", "git", "procps", "iproute2", "ca-certificates")
    .pip_install("uv==0.13.5")
    # The pod's exact package set, minus its editable install of the repo (a pod path). The
    # repo is put on PYTHONPATH at run time instead; bootstrap.sh also installs it editable.
    .add_local_file(FREEZE, "/opt/m1/pip-freeze.txt", copy=True)
    .run_commands(
        "grep -v '^-e ' /opt/m1/pip-freeze.txt > /opt/m1/requirements.txt",
        "uv venv /root/.venv --python 3.12",
        "uv pip install --python /root/.venv/bin/python --torch-backend=cu130"
        " -r /opt/m1/requirements.txt",
    )
    .env({
        "VENV": "/root/.venv",
        "WORKSPACE": "/workspace",
        # Each call is a new container; keep vLLM's compile cache on the Volume so a replica
        # start does not recompile every time.
        "VLLM_CACHE_ROOT": "/workspace/cache/vllm",
    })
    .add_local_dir(REPO, "/root/repo", ignore=_skip)
)


def _hf_secret() -> modal.Secret:
    """HF_TOKEN from bench/m1/.env, the file push.sh sends to the pod. Nothing else from it."""
    env = {}
    if modal.is_local() and (BENCH / ".env").is_file():
        for line in (BENCH / ".env").read_text().splitlines():
            k, sep, v = line.partition("=")
            if sep and k.strip() == "HF_TOKEN" and v.strip():
                env["HF_TOKEN"] = v.strip().strip("'\"")
    return modal.Secret.from_dict(env)


app = modal.App("manifold-m1")
volume = modal.Volume.from_name(VOLUME, create_if_missing=True)
common = dict(image=image, volumes={"/workspace": volume}, secrets=[_hf_secret()],
              timeout=TIMEOUT_S)


def _run(cmd: str, git_sha: str) -> None:
    m1 = Path("/workspace/m1")
    (m1 / "logs").mkdir(parents=True, exist_ok=True)
    # A fresh container has no replicas, router or tmux server. A current.env left by an earlier
    # call (or a preempted one) would make calibrate.sh skip starting the replicas.
    (m1 / "current.env").unlink(missing_ok=True)
    (Path("/root/repo") / "GIT_SHA").write_text(git_sha + "\n")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = Path(shlex.split(cmd)[0]).stem if cmd.strip() else "shell"
    logf = m1 / "logs" / f"modal-{stamp}-{name}.log"
    env = {**os.environ, "PYTHONPATH": "/root/repo"}
    print(f"[modal] {cmd}\n[modal] log: {logf}", flush=True)
    script = f"set -o pipefail; {{ {cmd} ; }} 2>&1 | tee -a {shlex.quote(str(logf))}"
    rc = subprocess.run(["bash", "-c", script], cwd="/root/repo/bench/m1", env=env).returncode
    volume.commit()
    if rc != 0:
        raise SystemExit(f"command exited {rc}: {cmd}")


@app.function(gpu=GPU, cpu=CPU, memory=MEMORY_MIB, **common)
def bench(cmd: str, git_sha: str) -> None:
    _run(cmd, git_sha)


# No GPU: the laptop dry run against manifold.sim, and quick checks of the image and Volume.
@app.function(cpu=4, memory=8 * 1024, **common)
def bench_cpu(cmd: str, git_sha: str) -> None:
    _run(cmd, git_sha)


def _git_sha() -> str:
    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    sha = git("rev-parse", "HEAD") or "unknown"
    return f"{sha}-dirty" if git("status", "--porcelain") else sha


@app.local_entrypoint()
def main(cmd: str, cpu: bool = False) -> None:
    """CMD runs with bash in bench/m1, e.g. "./bringup.sh q35-9b && ./calibrate.sh calib"."""
    (bench_cpu if cpu else bench).remote(cmd, _git_sha())
