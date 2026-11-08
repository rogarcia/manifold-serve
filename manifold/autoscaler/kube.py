"""Engines as pods. `vllm-core` holds slot 0 and `vllm-burst` holds slots 1 and up.

A StatefulSet removes its highest ordinal first, so the slot the policy drains is the pod
Kubernetes deletes. Pods and scale are read back on every call; nothing is remembered here.
"""

from __future__ import annotations

import ssl
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from .policy import Phase

API_URL = "https://kubernetes.default.svc"
SA_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
CORE, BURST = "vllm-core", "vllm-burst"
SELECTOR = "app.kubernetes.io/name=vllm"
MERGE_PATCH = {"content-type": "application/merge-patch+json"}


def _phase(pod: dict[str, Any]) -> Phase:
    if pod["metadata"].get("deletionTimestamp"):
        return Phase.STOPPING
    status = pod["status"]
    if status.get("phase") in ("Succeeded", "Failed"):
        return Phase.OFF  # an evicted pod is a record; the controller replaces it
    ready = any(c["type"] == "Ready" and c["status"] == "True"
                for c in status.get("conditions", ()))
    return Phase.READY if ready else Phase.STARTING


def _bearer(token_file: Path) -> Callable[[httpx.Request], httpx.Request]:
    def auth(request: httpx.Request) -> httpx.Request:
        # Projected tokens rotate on disk, so a token cached at start would expire.
        request.headers["authorization"] = f"Bearer {token_file.read_text().strip()}"
        return request

    return auth


class KubeFleet:
    def __init__(self, client: httpx.AsyncClient, namespace: str, slots: int):
        self.client, self.namespace = client, namespace
        self.pods = (f"{CORE}-0", *(f"{BURST}-{i}" for i in range(slots - 1)))

    @classmethod
    def in_cluster(cls, slots: int, sa_dir: Path = SA_DIR,
                   transport: httpx.AsyncBaseTransport | None = None) -> KubeFleet:
        client = httpx.AsyncClient(
            base_url=API_URL, timeout=5.0, transport=transport,
            verify=ssl.create_default_context(cafile=str(sa_dir / "ca.crt")),
            auth=_bearer(sa_dir / "token"),
        )
        return cls(client, (sa_dir / "namespace").read_text().strip(), slots)

    async def observe(self) -> tuple[Phase, ...]:
        resp = await self.client.get(f"/api/v1/namespaces/{self.namespace}/pods",
                                     params={"labelSelector": SELECTOR})
        resp.raise_for_status()
        found = {p["metadata"]["name"]: _phase(p) for p in resp.json()["items"]}
        return tuple(found.get(name, Phase.OFF) for name in self.pods)

    async def scale(self, n: int) -> None:
        for name, replicas in ((CORE, min(n, 1)), (BURST, max(n - 1, 0))):
            url = f"/apis/apps/v1/namespaces/{self.namespace}/statefulsets/{name}/scale"
            resp = await self.client.get(url)
            resp.raise_for_status()
            # The Scale object omits replicas when they are zero.
            if resp.json()["spec"].get("replicas", 0) != replicas:
                resp = await self.client.patch(url, json={"spec": {"replicas": replicas}},
                                               headers=MERGE_PATCH)
                resp.raise_for_status()
