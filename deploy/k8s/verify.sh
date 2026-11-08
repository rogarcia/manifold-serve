#!/usr/bin/env bash
# Runs the sim overlay on a throwaway kind cluster and checks the scaling path end to end.
# Needs a running Docker, kind and kubectl. Takes about five minutes. Run from anywhere:
#
#   deploy/k8s/verify.sh
#
# The cluster is deleted on every exit, pass or fail.
set -euo pipefail

CLUSTER=manifold-verify
IMAGE=manifold-serve:dev
PORT=18000
NS=manifold
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
WORK=$(mktemp -d)

kc() { kubectl --context "kind-$CLUSTER" -n "$NS" "$@"; }
say() { echo "== $*"; }
die() { echo "FAIL: $*" >&2; exit 1; }

cleanup() {
  status=$?
  if [ "$status" -ne 0 ]; then
    kc get pods -o wide || true
    kc logs deploy/autoscaler --tail=40 || true
  fi
  jobs -p | xargs kill 2>/dev/null || true
  kind delete cluster --name "$CLUSTER" || true
  /bin/rm -rf "$WORK"
  exit "$status"
}
trap cleanup EXIT

until_true() {
  local deadline=$((SECONDS + $1)) what=$2
  shift 2
  until "$@"; do
    [ "$SECONDS" -lt "$deadline" ] || die "timed out waiting for: $what"
    sleep 1
  done
}

replicas() { kc get statefulset "$1" -o jsonpath='{.spec.replicas}'; }
engines() { kc get pods -l app.kubernetes.io/name=vllm -o name; }
no_engines() { [ -z "$(engines)" ]; }
pod_ready() {
  [ "$(kc get pod "$1" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)" = True ]
}
slot1_draining() {
  curl -fsS "localhost:$PORT/stats" | grep -q '"index":1,"url":"[^"]*","state":"draining"'
}
chat() {
  local headers
  headers=$(mktemp "$WORK/headers.XXXXXX")
  curl -sS --max-time 200 -o /dev/null -D "$headers" \
    -w '%{http_code} %{time_total}' "localhost:$PORT/v1/chat/completions" \
    -H 'content-type: application/json' \
    -d "{\"model\": \"sim-model\", \"max_tokens\": $1, \"messages\": [{\"role\": \"user\", \"content\": \"hello\"}]}"
  printf ' %s\n' "$(tr -d '\r' < "$headers" | awk 'tolower($1) == "x-manifold-backend:" {print $2}')"
}

for tool in docker kind kubectl curl; do
  command -v "$tool" >/dev/null || die "$tool is not installed"
done

say "cluster and image"
kind create cluster --name "$CLUSTER" --wait 120s
docker build -f "$ROOT/deploy/k8s/Dockerfile" -t "$IMAGE" "$ROOT"
kind load docker-image "$IMAGE" --name "$CLUSTER"

say "apply the sim overlay"
kubectl --context "kind-$CLUSTER" apply -k "$ROOT/deploy/k8s/sim"
kc rollout status deployment/router --timeout=120s
kc rollout status deployment/autoscaler --timeout=120s

say "the autoscaler's service account can do what it needs and no more"
SA=system:serviceaccount:$NS:autoscaler
kc auth can-i list pods --as "$SA" >/dev/null || die "cannot list pods"
kc auth can-i patch statefulsets/vllm-core --subresource=scale --as "$SA" >/dev/null \
  || die "cannot patch vllm-core scale"
if kc auth can-i delete pods --as "$SA" >/dev/null 2>&1; then die "can delete pods"; fi
if kc auth can-i patch statefulsets/vllm-core --as "$SA" >/dev/null 2>&1; then
  die "can patch the whole StatefulSet"
fi

kubectl --context "kind-$CLUSTER" -n "$NS" port-forward service/router "$PORT:8000" \
  >"$WORK/port-forward.log" 2>&1 &
until_true 30 "port-forward" curl -fsS -o /dev/null "localhost:$PORT/healthz"

say "the fleet rests at zero"
no_engines || die "engines exist before any demand"

say "zero to one: a held request wakes slot 0 and is served"
read -r status seconds backend < <(chat 8)
[ "$status" = 200 ] || die "first request returned $status"
case $backend in *vllm-core-0*) ;; *) die "first request served by '$backend'" ;; esac
[ "${seconds%.*}" -ge 15 ] || die "first request took $seconds s, so it was not held for the start"
echo "held for $seconds s, served by $backend"

say "load adds slot 1"
for _ in 1 2 3 4 5 6; do
  chat 3000 >/dev/null &
done
until_true 90 "vllm-burst-0 Ready" pod_ready vllm-burst-0

say "a deleted burst pod drains and comes back"
old_uid=$(kc get pod vllm-burst-0 -o jsonpath='{.metadata.uid}')
kc delete pod vllm-burst-0 --wait=false
until_true 10 "router marks slot 1 draining" slot1_draining
new_burst() {
  uid=$(kc get pod vllm-burst-0 -o jsonpath='{.metadata.uid}' 2>/dev/null) \
    && [ "$uid" != "$old_uid" ] && pod_ready vllm-burst-0
}
until_true 90 "a new vllm-burst-0 Ready" new_burst

say "no demand scales the fleet back to zero"
until_true 180 "load to finish and all engines to go" no_engines
[ "$(replicas vllm-core)" = 0 ] || die "vllm-core is not at zero"
[ "$(replicas vllm-burst)" = 0 ] || die "vllm-burst is not at zero"

say "PASS"
