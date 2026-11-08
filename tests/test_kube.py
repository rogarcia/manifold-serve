"""KubeFleet against a fake API server, and consistency checks on the rendered manifests.

The server below is a fake written from the documented API shapes. It proves no real API server
behaviour: passing here shows only that KubeFleet does what its author believes the API wants.
`deploy/k8s/verify.sh` is the check against a real API server.
"""

import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path

import certifi
import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from manifold.autoscaler.__main__ import run
from manifold.autoscaler.kube import KubeFleet
from manifold.autoscaler.policy import Phase, Signal, Spec
from manifold.router.app import Config
from manifold.router.replicas import ReplicaState

NS = "manifold"
LABELS = {"app.kubernetes.io/name": "vllm"}
ROOT = Path(__file__).resolve().parent.parent


def pod(name, phase="Running", ready=False, deleting=False, labels=LABELS):
    meta = {"name": name, "namespace": NS, "labels": labels}
    if deleting:
        meta["deletionTimestamp"] = "2026-11-07T00:00:00Z"
    status = {"phase": phase}
    if phase == "Running":
        status["conditions"] = [{"type": "Ready", "status": "True" if ready else "False"}]
    return {"metadata": meta, "status": status}


class FakeCluster:
    def __init__(self):
        self.replicas = {"vllm-core": 0, "vllm-burst": 0}
        self.pods = []
        self.patches = []
        self.auth = []
        self.app = app = FastAPI()

        @app.get("/api/v1/namespaces/{ns}/pods")
        async def pods(ns: str, request: Request) -> dict:
            self.auth.append(request.headers.get("authorization"))
            key, _, value = request.query_params["labelSelector"].partition("=")
            return {"kind": "PodList", "items": [
                p for p in self.pods
                if ns == p["metadata"]["namespace"] and p["metadata"]["labels"].get(key) == value
            ]}

        def scale_body(name: str) -> dict:
            n = self.replicas[name]
            spec = {"replicas": n} if n else {}  # the real Scale omits a zero
            return {"kind": "Scale", "metadata": {"name": name}, "spec": spec}

        @app.get("/apis/apps/v1/namespaces/{ns}/statefulsets/{name}/scale")
        async def get_scale(ns: str, name: str):
            if ns != NS or name not in self.replicas:
                return JSONResponse({"kind": "Status", "code": 404}, 404)
            return scale_body(name)

        @app.patch("/apis/apps/v1/namespaces/{ns}/statefulsets/{name}/scale")
        async def patch_scale(ns: str, name: str, request: Request):
            if request.headers.get("content-type") != "application/merge-patch+json":
                return JSONResponse({"kind": "Status", "code": 415}, 415)
            n = (await request.json())["spec"]["replicas"]
            self.patches.append((name, n))
            self.replicas[name] = n
            return scale_body(name)


@pytest.fixture
async def cluster():
    c = FakeCluster()
    c.client = httpx.AsyncClient(transport=httpx.ASGITransport(c.app), base_url="http://kube")
    yield c
    await c.client.aclose()


@pytest.mark.parametrize(("spec", "phase"), [
    ({"phase": "Pending"}, Phase.STARTING),
    ({"ready": False}, Phase.STARTING),
    ({"ready": True}, Phase.READY),
    ({"ready": True, "deleting": True}, Phase.STOPPING),
    ({"phase": "Pending", "deleting": True}, Phase.STOPPING),
    ({"phase": "Failed"}, Phase.OFF),
    ({"phase": "Succeeded"}, Phase.OFF),
])
async def test_pod_state_becomes_a_phase(cluster, spec, phase):
    cluster.pods = [pod("vllm-burst-0", **spec)]
    fleet = KubeFleet(cluster.client, NS, slots=2)
    assert await fleet.observe() == (Phase.OFF, phase)


async def test_ordinals_map_to_slots_and_absent_slots_are_off(cluster):
    cluster.pods = [pod("vllm-burst-1", ready=True), pod("vllm-core-0", ready=True)]
    fleet = KubeFleet(cluster.client, NS, slots=4)
    assert await fleet.observe() == (Phase.READY, Phase.OFF, Phase.READY, Phase.OFF)


async def test_only_labelled_pods_in_slot_range_count(cluster):
    cluster.pods = [
        pod("vllm-burst-0", ready=True, labels={"app.kubernetes.io/name": "other"}),
        pod("vllm-burst-1", ready=True),
        pod("vllm-core-1", ready=True),
        pod("stray", ready=True),
    ]
    fleet = KubeFleet(cluster.client, NS, slots=2)
    assert await fleet.observe() == (Phase.OFF, Phase.OFF)


@pytest.mark.parametrize(("n", "core", "burst"), [(0, 0, 0), (1, 1, 0), (2, 1, 1), (4, 1, 3)])
async def test_scale_splits_slots_across_the_two_statefulsets(cluster, n, core, burst):
    cluster.replicas = {"vllm-core": 1, "vllm-burst": 2}
    await KubeFleet(cluster.client, NS, slots=4).scale(n)
    assert cluster.replicas == {"vllm-core": core, "vllm-burst": burst}


async def test_scale_patches_only_what_differs(cluster):
    fleet = KubeFleet(cluster.client, NS, slots=2)
    await fleet.scale(0)
    assert cluster.patches == []
    await fleet.scale(2)
    assert sorted(cluster.patches) == [("vllm-burst", 1), ("vllm-core", 1)]
    await fleet.scale(2)
    assert len(cluster.patches) == 2
    await fleet.scale(1)
    assert cluster.patches[2:] == [("vllm-burst", 0)]


async def test_api_errors_reach_the_loop_as_http_errors(cluster):
    fleet = KubeFleet(cluster.client, "elsewhere", slots=2)
    with pytest.raises(httpx.HTTPError):
        await fleet.scale(1)


async def test_in_cluster_rereads_the_rotating_token(cluster, tmp_path):
    (tmp_path / "ca.crt").write_text(Path(certifi.where()).read_text())
    (tmp_path / "namespace").write_text(NS + "\n")
    (tmp_path / "token").write_text("first\n")
    fleet = KubeFleet.in_cluster(2, sa_dir=tmp_path, transport=httpx.ASGITransport(cluster.app))
    await fleet.observe()
    (tmp_path / "token").write_text("second\n")
    await fleet.observe()
    await fleet.client.aclose()
    assert cluster.auth == ["Bearer first", "Bearer second"]


class StubRouter:
    def __init__(self, demand: int):
        self.demand, self.pushed = demand, []

    async def signal(self) -> Signal:
        return Signal(interactive=self.demand, batch=0, in_flight=(0, 0))

    async def push(self, states: list) -> None:
        self.pushed.append(states)


async def test_the_loop_drives_a_slot_from_zero_to_ready(cluster):
    router = StubRouter(demand=1)
    fleet = KubeFleet(cluster.client, NS, slots=2)
    loop = asyncio.create_task(run(fleet, router, Spec(tick_s=0.001), None))

    async def until(reached) -> None:
        while not reached():
            await asyncio.sleep(0.001)

    try:
        await asyncio.wait_for(until(lambda: cluster.replicas["vllm-core"] == 1), 5)
        assert cluster.replicas == {"vllm-core": 1, "vllm-burst": 0}
        cluster.pods = [pod("vllm-core-0", ready=True)]
        served = lambda: router.pushed and router.pushed[-1][0] is ReplicaState.READY
        await asyncio.wait_for(until(served), 5)
    finally:
        loop.cancel()
    assert router.pushed[-1] == [ReplicaState.READY, ReplicaState.OFF]


def render(path: str) -> list[dict]:
    yaml = pytest.importorskip("yaml")
    if shutil.which("kubectl") is None:
        pytest.skip("kubectl not installed")
    out = subprocess.run(["kubectl", "kustomize", str(ROOT / path)], capture_output=True,
                         text=True, check=True).stdout
    return list(yaml.safe_load_all(out))


def find(objects: list[dict], kind: str, name: str | None = None) -> list[dict] | dict:
    found = [o for o in objects if o["kind"] == kind]
    if name is None:
        return found
    return next(o for o in found if o["metadata"]["name"] == name)


def container(workload: dict) -> dict:
    return workload["spec"]["template"]["spec"]["containers"][0]


def env_of(workload: dict) -> dict[str, str]:
    return {e["name"]: e["value"] for e in container(workload)["env"]}


@pytest.fixture(params=["deploy/k8s", "deploy/k8s/sim"], scope="module")
def rendered(request) -> list[dict]:
    return render(request.param)


def test_every_manifold_variable_set_is_one_the_code_reads(rendered):
    router_src = (ROOT / "manifold/router/app.py").read_text()
    loop_src = (ROOT / "manifold/autoscaler/__main__.py").read_text()
    pairs = [("router", router_src), ("autoscaler", loop_src)]
    for name, source in pairs:
        read = set(re.findall(r'env(?:\.get\(|\[)"(MANIFOLD_\w+)"', source))
        set_vars = set(env_of(find(rendered, "Deployment", name)))
        assert set_vars <= read, name


def test_router_config_parses_with_the_values_the_spec_asks_for(rendered):
    cfg = Config.from_env(env_of(find(rendered, "Deployment", "router")))
    assert cfg.host == "0.0.0.0"
    assert cfg.initial_state.value == "off"
    assert (cfg.caps.interactive, cfg.caps.batch) == (96, 64)


def test_backends_are_the_stable_names_of_every_slot(rendered):
    spec = Spec(**json.loads(env_of(find(rendered, "Deployment", "autoscaler"))["MANIFOLD_SPEC"]))
    backends = Config.from_env(env_of(find(rendered, "Deployment", "router"))).backends
    assert len(backends) == spec.max_replicas
    service = find(rendered, "Service", "vllm")["metadata"]
    pods = KubeFleet(None, NS, spec.max_replicas).pods
    assert backends == [
        f"http://{p}.{service['name']}.{service['namespace']}.svc.cluster.local:8000" for p in pods
    ]


def test_references_resolve(rendered):
    workloads = find(rendered, "Deployment") + find(rendered, "StatefulSet")
    priorities = {o["metadata"]["name"] for o in find(rendered, "PriorityClass")}
    accounts = {o["metadata"]["name"] for o in find(rendered, "ServiceAccount")}
    for w in workloads:
        pod_spec = w["spec"]["template"]["spec"]
        assert pod_spec["priorityClassName"] in priorities
        assert pod_spec.get("serviceAccountName", "autoscaler") in accounts
        assert w["metadata"]["namespace"] == NS

    binding = find(rendered, "RoleBinding", "autoscaler")
    assert find(rendered, "Role", binding["roleRef"]["name"])
    assert {s["name"] for s in binding["subjects"]} <= accounts

    url = env_of(find(rendered, "Deployment", "autoscaler"))["MANIFOLD_ROUTER_URL"]
    host = url.removeprefix("http://").split(".")
    router = find(rendered, "Service", host[0])
    assert host[1] == router["metadata"]["namespace"]
    assert router["spec"]["selector"].items() \
        <= find(rendered, "Deployment", "router")["spec"]["template"]["metadata"]["labels"].items()


def test_statefulsets_are_reachable_through_the_headless_service(rendered):
    service = find(rendered, "Service", "vllm")
    assert service["spec"]["clusterIP"] == "None"
    for name in ("vllm-core", "vllm-burst"):
        sts = find(rendered, "StatefulSet", name)
        assert sts["spec"]["serviceName"] == "vllm"
        assert sts["spec"]["replicas"] == 0
        labels = sts["spec"]["template"]["metadata"]["labels"]
        assert service["spec"]["selector"].items() <= labels.items()
        assert LABELS.items() <= labels.items()


def test_role_covers_the_calls_the_fleet_makes_and_nothing_else(rendered):
    role = find(rendered, "Role", "autoscaler")
    rules = {(r["resources"][0], tuple(r["verbs"])) for r in role["rules"]}
    assert rules == {("pods", ("get", "list")), ("statefulsets/scale", ("get", "patch"))}


def test_both_tiers_run_the_same_engine_and_only_burst_is_spot_tolerant(rendered):
    core = find(rendered, "StatefulSet", "vllm-core")
    burst = find(rendered, "StatefulSet", "vllm-burst")
    assert container(core) == container(burst)
    assert "tolerations" not in core["spec"]["template"]["spec"]
    tolerations = burst["spec"]["template"]["spec"]["tolerations"]
    assert [(t["key"], t["value"], t["effect"]) for t in tolerations] \
        == [("manifold.io/spot", "true", "NoSchedule")]
    values = {o["metadata"]["name"]: o["value"] for o in find(rendered, "PriorityClass")}
    assert values["manifold-serving"] > values["manifold-burst"]


def test_an_engine_can_start_and_drain_inside_its_limits():
    objects = render("deploy/k8s")
    sts = find(objects, "StatefulSet", "vllm-core")
    c = container(sts)
    startup = c["startupProbe"]
    assert startup["periodSeconds"] * startup["failureThreshold"] >= 300
    assert c["readinessProbe"]["httpGet"]["path"] == "/health"
    assert c["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert c["image"] == "vllm/vllm-openai:v0.31.0"
    args = c["args"]
    shutdown = int(args[args.index("--shutdown-timeout") + 1])
    pre_stop = int(c["lifecycle"]["preStop"]["exec"]["command"][1])
    assert pre_stop + shutdown < sts["spec"]["template"]["spec"]["terminationGracePeriodSeconds"]


def test_engine_flags_match_the_compose_file():
    compose = (ROOT / "deploy/docker-compose.yml").read_text()
    args = container(find(render("deploy/k8s"), "StatefulSet", "vllm-core"))["args"]
    block = compose[compose.index("x-args:"):compose.index("services:")].split("\n", 1)[1]
    assert " ".join(block.split()).replace("'", "") == " ".join(a for a in args[:-2])


def test_sim_overlay_runs_without_gpus_or_vllm():
    objects = render("deploy/k8s/sim")
    assert "nvidia.com/gpu" not in json.dumps(objects)
    assert "vllm/vllm-openai" not in json.dumps(objects)
    sts = find(objects, "StatefulSet", "vllm-burst")
    assert "volumeClaimTemplates" not in sts["spec"]
    c = container(sts)
    assert c["command"] == ["python", "-m", "manifold.sim"]
    assert "--start-delay-s" in c["args"]
    assert "--shutdown-timeout-s" in c["args"]
    assert c["lifecycle"]["preStop"] == container(find(render("deploy/k8s"), "StatefulSet",
                                                       "vllm-burst"))["lifecycle"]["preStop"]
