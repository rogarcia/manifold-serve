# Kubernetes deployment

These manifests run the same control loop that was measured on Modal. The router holds requests,
`python -m manifold.autoscaler` decides, and `KubeFleet` applies the answer to two StatefulSets.

| object | purpose |
|---|---|
| `vllm-core` | slot 0, high priority, not spot-tolerant |
| `vllm-burst` | slots 1 and up, low priority, tolerates `manifold.io/spot=true:NoSchedule` |
| `vllm` | headless Service that gives each slot a stable DNS name |
| `router` | one replica, because the gate and the prefix index live in memory |
| `autoscaler` | one replica, allowed to list pods and to scale these two StatefulSets |

Both StatefulSets start at zero replicas. Slot `i` in `MANIFOLD_BACKENDS` is slot `i` in
`KubeFleet`, and `MANIFOLD_BACKENDS` must list exactly `max_replicas` slots. To add a slot, extend
the list and raise `max_replicas` in `MANIFOLD_SPEC`.

## Layout

`base/` holds the objects. `kustomization.yaml` here points at it, and `sim/` overlays it, because
kustomize refuses an overlay that sits below its own base.

    kubectl kustomize deploy/k8s        # render for GPUs
    kubectl kustomize deploy/k8s/sim    # render for a laptop

## Running it on a laptop

`deploy/k8s/verify.sh` needs a running Docker, kind and kubectl. It creates a cluster, builds and
loads the image, and applies the `sim/` overlay. Then it checks these things in order:

1. The autoscaler's ServiceAccount can list pods and scale the StatefulSets, and cannot delete pods
   or patch a whole StatefulSet.
2. No engine pod exists before any demand.
3. A first request is held through the engine's start and returns 200 from `vllm-core-0`.
4. Six concurrent requests bring up `vllm-burst-0`.
5. Deleting `vllm-burst-0` turns slot 1 to `draining` on the router, and the controller brings
   back a new pod.
6. With no demand left, both StatefulSets return to zero and every engine pod is gone.

The cluster is deleted on every exit.

The overlay replaces vLLM with `python -m manifold.sim`, drops the GPU and the model cache volume,
and shortens the timers. A sim start takes 15 s because `--start-delay-s` is kept, so the probes
have a real starting phase to see.

## What this proves and what it does not

`tests/test_kube.py` runs `KubeFleet` against a fake API server written from the documented API
shapes. It shows that the adapter does what its author believes the API wants. It proves nothing
about a real API server. Its rendered-manifest checks show that the names, the environment
variables and the slot count agree with each other and with the code.

A passing `verify.sh` would show that a real API server accepts the manifests and the RBAC, that
pods, probes, SIGTERM and DNS behave as the loop expects, and that a real StatefulSet controller
follows `KubeFleet`.

No part of this has shown the following:

- vLLM on Kubernetes. The overlay runs the sim, and the vLLM manifests have not been applied
  anywhere. The image name and flags match the Docker Compose file, with `--shutdown-timeout 25`
  added.
- GPU scheduling. kind has no `nvidia.com/gpu` resource, and a real cluster needs the device plugin.
- PriorityClass eviction. Nothing in `verify.sh` makes a pod compete for a resource.
- Spot reclaim and node provisioning. Deleting a pod is a stand-in for the path after a reclaim
  notice. A lost node can also leave a pod terminating until the node object goes away, and the
  slot then stays `stopping`, because a StatefulSet will not replace a pod it cannot confirm is gone.
- Any latency, throughput or cost number. The measurements come from Modal.

## Before using a real cluster

- Build the image, push it, and point the manifests at the pushed name with a kustomize `images:`
  entry. `manifold-serve:dev` exists only where it was built.
- The toleration lets burst pods onto spot nodes and does not attract them there. Add node affinity
  if burst must prefer spot.
- Each engine pod claims a 50 GiB volume for the Hugging Face cache, so a restarted slot does not
  download the model again. The claims outlive scale-to-zero and are billed while idle. This needs
  a default StorageClass.
- `/admin` on the router has no authentication. The Service is ClusterIP only, and anything that
  can reach it can change slot state. `x-manifold-class` also needs a trusted gateway in front.
- `VLLM_SERVER_DEV_MODE` is not set, unlike the Compose file, because it exposes dev endpoints and
  every pod in the cluster can reach an engine's port.
- `kubectl apply` resets both StatefulSets to the zero replicas in the manifests. The autoscaler
  patches them back within a tick, but pods may already be terminating, so apply changes when the
  fleet is idle.
- A sim pod takes the full 35 s to stop if it is deleted during `--start-delay-s`. Python runs as
  PID 1 and sleeps before it installs a SIGTERM handler. vLLM has not been checked for this.

## KEDA

The router exports `manifold_router_demand_requests{class,stage}`, which exists at zero replicas.
KEDA can use it without any change to the router, given a Prometheus that scrapes `router:8000`.
One ScaledObject per StatefulSet gives the same core and burst split:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-core
  namespace: manifold
spec:
  scaleTargetRef:
    kind: StatefulSet
    name: vllm-core
  minReplicaCount: 0
  maxReplicaCount: 1
  cooldownPeriod: 170
  triggers:
    - type: prometheus
      metadata:
        serverAddress: http://prometheus.monitoring:9090
        query: sum(manifold_router_demand_requests)
        threshold: "26"
        activationThreshold: "0"
```

This example has not been run, and it is not part of the kustomization.

The loop is used instead for two reasons. The policy that decides has to be the code that was
measured, and KEDA cannot run on Modal, so a KEDA deployment would ship a policy that was never
tested. The loop also owns the drain: it holds a slot in `draining` until the slot is empty or
`drain_deadline_s` passes, and only then lowers the replica count. StatefulSet removal of the
highest ordinal matches that order, because the policy always drains the top slot. An HPA changes
the count first and leaves the drain to whatever is running in the pod.
