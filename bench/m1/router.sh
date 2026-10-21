#!/usr/bin/env bash
# Restart the router in tmux session "router" with POLICY, logging into CELL_DIR, and wait for
# /healthz. `router.sh stop` stops it with SIGINT so the decision log is flushed.
# Usage: router.sh POLICY CELL_DIR | router.sh stop
#   ROUTER_BACKENDS (default "$URL_A,$URL_B") overrides the replica list, e.g. A only.
set -euo pipefail
source "$(dirname "$0")/env.sh"
source "$BENCH_DIR/lib.sh"

POLICY=${1:?usage: router.sh POLICY CELL_DIR | router.sh stop}

router_stop() {
  tmux has-session -t router 2>/dev/null || return 0
  tmux send-keys -t router C-c
  for _ in $(seq 1 30); do
    tmux has-session -t router 2>/dev/null || return 0
    sleep 0.5
  done
  log "router ignored SIGINT for 15 s; killing the session"
  tmux kill-session -t router 2>/dev/null || true
}

if [ "$POLICY" = stop ]; then router_stop; exit 0; fi
case "$POLICY" in round_robin|least_load|session_hash|prefix_aware) ;; *) die "unknown policy: $POLICY" ;; esac
CELL_DIR=${2:?usage: router.sh POLICY CELL_DIR}
load_current
mkdir -p "$CELL_DIR"
router_stop

# The router's prefix index is sized to one replica's KV pool, in blocks.
KV_MIN=$(( CUR_KV_TOKENS_A < CUR_KV_TOKENS_B ? CUR_KV_TOKENS_A : CUR_KV_TOKENS_B ))
INDEX_BLOCKS=$(( KV_MIN / CUR_BLOCK_TOKENS > 0 ? KV_MIN / CUR_BLOCK_TOKENS : 1 ))
# The sim tokenizes UTF-8 bytes; a pod run tokenizes with the served model's pinned snapshot.
if [ "$CUR_SIM" = 1 ]; then TOK=${ROUTER_TOKENIZER:-bytes}; else TOK=${ROUTER_TOKENIZER:-$TOKENIZER_PATH}; fi

# Stdin is a tty inside tmux; the C-c from router_stop reaches python as SIGINT.
launch router "$CELL_DIR/router.log" env \
  MANIFOLD_BACKENDS="${ROUTER_BACKENDS:-$URL_A,$URL_B}" \
  MANIFOLD_POLICY="$POLICY" \
  MANIFOLD_HOST=127.0.0.1 \
  MANIFOLD_PORT="$ROUTER_PORT" \
  MANIFOLD_TOKENIZER="$TOK" \
  MANIFOLD_CHAT_TEMPLATE_KWARGS="$CHAT_TEMPLATE_KWARGS" \
  MANIFOLD_BLOCK_TOKENS="$CUR_BLOCK_TOKENS" \
  MANIFOLD_INDEX_BLOCKS="$INDEX_BLOCKS" \
  MANIFOLD_DECISION_LOG="$CELL_DIR/decisions.jsonl" \
  python -m manifold.router
wait_http "$ROUTER_URL/healthz" 180 router "$CELL_DIR/router.log"
log "router up: $POLICY, block $CUR_BLOCK_TOKENS, index $INDEX_BLOCKS blocks, tokenizer $TOK"
