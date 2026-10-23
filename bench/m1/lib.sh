# shellcheck shell=bash
# Helpers shared by the bench/m1 scripts. Source env.sh first.

log() { echo "[$(date -u +%H:%M:%S)] $*" >&2; }
die() { log "FATAL: $*"; exit 1; }

# Start CMD... detached in tmux session SESSION, output to LOG. The command runs from a generated
# script under $M1/run/ that re-exports LAUNCH_VARS and re-sources env.sh, because tmux gives a new
# session the tmux server's environment, not the caller's.
launch() {
  local session=$1 logf=$2; shift 2
  local script=$M1/run/$session.sh v
  mkdir -p "$M1/run" "$(dirname "$logf")"
  {
    echo '#!/usr/bin/env bash'
    for v in $LAUNCH_VARS; do
      if [ -n "${!v+x}" ]; then printf 'export %s=%q\n' "$v" "${!v}"; fi
    done
    printf 'source %q\n' "$BENCH_DIR/env.sh"
    printf 'exec'; printf ' %q' "$@"; echo
  } > "$script"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" "bash $(printf %q "$script") > $(printf %q "$logf") 2>&1"
}

# Fail unless nothing listens on PORT. curl exits 7 only when the connection is refused; any
# HTTP answer means another process (RunPod's nginx, a stale server) would answer our health checks.
require_free_port() {
  local rc=0
  curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$1/" || rc=$?
  [ "$rc" = 7 ] || die "port $1 is already in use (see: ss -ltnp | grep :$1)"
}

# Wait until URL answers 200. Fails early if tmux SESSION exits, printing the tail of LOG.
wait_http() {
  local url=$1 timeout_s=$2 session=${3:-} logf=${4:-} t0=$SECONDS
  while (( SECONDS - t0 < timeout_s )); do
    curl -sf --max-time 5 "$url" >/dev/null && return 0
    if [ -n "$session" ] && ! tmux has-session -t "$session" 2>/dev/null; then
      log "$session exited before $url answered; tail of $logf:"
      tail -40 "$logf" >&2 || true
      return 1
    fi
    sleep 2
  done
  log "timed out after ${timeout_s}s waiting for $url"
  [ -n "$logf" ] && tail -40 "$logf" >&2
  return 1
}

scrape() { curl -sf --max-time 10 "$1/metrics"; }

# Sum of every sample of metric NAME (all label sets) in Prometheus text on stdin.
metric_sum() {
  awk -v n="$1" '$1 == n || index($1, n "{") == 1 { s += $NF } END { printf "%.0f\n", s }'
}

# Wait until both replicas report no running and no waiting requests.
wait_idle() {
  local timeout_s=${1:-180} t0=$SECONDS url m busy
  while (( SECONDS - t0 < timeout_s )); do
    busy=0
    for url in "$URL_A" "$URL_B"; do
      m=$(scrape "$url") || { busy=1; continue; }
      (( $(metric_sum vllm:num_requests_running <<< "$m") + $(metric_sum vllm:num_requests_waiting <<< "$m") > 0 )) && busy=1
    done
    [ "$busy" = 0 ] && return 0
    sleep 1
  done
  log "replicas still busy after ${timeout_s}s"; return 1
}

reset_prefix_cache() { curl -sf --max-time 30 -X POST "$1/reset_prefix_cache" >/dev/null || die "reset_prefix_cache failed on $1"; }

# POST one chat completion straight to URL. Prints the response JSON.
chat_once() {
  local url=$1 content=$2 max_tokens=$3
  python -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "messages": [{"role": "user", "content": sys.argv[2]}], "max_tokens": int(sys.argv[3]), "temperature": 0}))' \
    "$MODEL" "$content" "$max_tokens" \
    | curl -sf --max-time 300 -H 'Content-Type: application/json' -d @- "$url/v1/chat/completions"
}

# Send one prompt twice to URL and require vllm:prefix_cache_hits_total to rise on the second
# pass. A hybrid model only caches whole blocks (1,568 tokens on M0's model), so the prompt spans
# about 2.5 blocks, capped at a quarter of the smaller KV pool so a small pool can hold it.
# Needs current.env loaded.
canary() {
  local url=$1 block=$CUR_BLOCK_TOKENS text h1 h2 t0 target pool
  pool=$(( CUR_KV_TOKENS_A < CUR_KV_TOKENS_B ? CUR_KV_TOKENS_A : CUR_KV_TOKENS_B ))
  target=$(( block * 5 / 2 )); (( target <= pool / 4 )) || target=$(( pool / 4 ))
  # About 14 tokens per sentence.
  text=$(python -c 'import sys; n=int(sys.argv[1]); print(" ".join(f"Canary sentence {i} checks that the prefix cache stores this prompt." for i in range(n)))' $(( target / 14 + 1 )))
  chat_once "$url" "$text" 1 >/dev/null || { log "canary request 1 failed on $url"; return 1; }
  h1=$(scrape "$url" | metric_sum vllm:prefix_cache_hits_total)
  chat_once "$url" "$text" 1 >/dev/null || { log "canary request 2 failed on $url"; return 1; }
  t0=$SECONDS
  while (( SECONDS - t0 < 10 )); do
    h2=$(scrape "$url" | metric_sum vllm:prefix_cache_hits_total)
    (( h2 > h1 )) && { log "canary ok on $url: prefix_cache_hits_total $h1 -> $h2"; return 0; }
    sleep 1
  done
  log "canary FAILED on $url: prefix_cache_hits_total stayed at $h1"; return 1
}

# Facts from one vLLM (or manifold.sim) startup log, as KEY=VALUE lines.
parse_serve_log() {
  local f=$1 v
  v=$(grep -oE 'Setting attention block size to [0-9]+ tokens' "$f" | tail -1 | grep -oE '[0-9]+' || true)
  if [ -n "$v" ]; then echo "BLOCK_TOKENS=$v"; echo "BLOCK_SOURCE=log"
  else echo "BLOCK_TOKENS=16"; echo "BLOCK_SOURCE=default"; fi
  v=$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$f" | tail -1 | grep -oE '[0-9][0-9,]*' | tr -d , || true)
  echo "KV_TOKENS=${v:-}"
  v=$(grep -oE 'Available KV cache memory: [0-9.]+ GiB' "$f" | tail -1 | grep -oE '[0-9]+\.?[0-9]*' || true)
  echo "KV_GIB=${v:-}"
  v=$(grep -oE "cudagraph_capture_sizes'?: \[[0-9, ]*\]" "$f" | tail -1 | grep -oE '\[[0-9, ]*\]' | tr -d ' ' || true)
  echo "CUDAGRAPH_SIZES=${v:-}"
}

# Run CMD... with a wall-clock limit; returns 124 on timeout like coreutils timeout(1).
with_timeout() {
  local secs=$1; shift
  "$@" & local pid=$!
  ( sleep "$secs"; kill -TERM "$pid" 2>/dev/null; sleep 10; kill -KILL "$pid" 2>/dev/null ) & local dog=$!
  local rc=0
  wait "$pid" || rc=$?
  kill "$dog" 2>/dev/null; wait "$dog" 2>/dev/null || true
  if (( rc == 143 || rc == 137 )); then return 124; fi
  return "$rc"
}

load_current() {
  [ -f "$M1/current.env" ] || die "no $M1/current.env: start the replicas with serve.sh first"
  . "$M1/current.env"
}
