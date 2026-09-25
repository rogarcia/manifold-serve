#!/usr/bin/env bash
# Run one benchmark cell against the running server.
# Usage: bench.sh CELL CONCURRENCY [SAMPLING] [THINKING]
#   SAMPLING: greedy (default) | qwen   (Qwen non-thinking: T=0.7 top_p=0.8 top_k=20)
#   THINKING: off (default) | on
# Result JSON lands in $M0/results/<config>/<cell>-c<conc>-<sampling>-think<on|off>.json
# With RUN set (e.g. RUN=2026-09-24), it lands in $M0/results/$RUN/<config>/ and the name
# gets -s<seed>, so repeats with other seeds do not overwrite each other.
# OUTPUT_LEN overrides the per-cell output length (e.g. for thinking-on cells).
set -euo pipefail
source "$(dirname "$0")/env.sh"

CELL=${1:?cell name}; CONC=${2:?concurrency}; SAMPLING=${3:-greedy}; THINKING=${4:-off}
CONFIG=$(cat "$M0/current-config")
DATA=$M0/data
N=${NUM_PROMPTS:-80}

case "$CELL" in
  chat)      DS=(--dataset-name hf --dataset-path philschmid/mt-bench --hf-output-len ${OUTPUT_LEN:-512}) ;;
  code)      DS=(--dataset-name hf --dataset-path likaixin/InstructCoder --hf-output-len ${OUTPUT_LEN:-512}) ;;
  math)      DS=(--dataset-name hf --dataset-path openai/gsm8k --hf-subset main --hf-split test --hf-output-len ${OUTPUT_LEN:-512}) ;;
  summarize) DS=(--dataset-name spec_bench --dataset-path "$DATA/spec_bench.jsonl" --spec-bench-category summarization --spec-bench-output-len 512) ;;
  translate) DS=(--dataset-name spec_bench --dataset-path "$DATA/spec_bench.jsonl" --spec-bench-category translation --spec-bench-output-len 512) ;;
  rag)       DS=(--dataset-name spec_bench --dataset-path "$DATA/spec_bench.jsonl" --spec-bench-category rag --spec-bench-output-len 512) ;;
  qa)        DS=(--dataset-name spec_bench --dataset-path "$DATA/spec_bench.jsonl" --spec-bench-category qa --spec-bench-output-len 512) ;;
  long1k|long8k|long16k|long32k)
             # SPEED_CAT: low_entropy | mixed | high_entropy (unset = all three mixed together)
             DS=(--dataset-name speed_bench --dataset-path "$DATA/speed_bench" --speed-bench-dataset-subset "throughput_${CELL#long}" --speed-bench-output-len 1024 ${SPEED_CAT:+--speed-bench-category "$SPEED_CAT"}); N=${NUM_PROMPTS:-32} ;;
  random)    DS=(--dataset-name random --random-input-len 1024 --random-output-len 512) ;;  # control only
  *) echo "unknown cell: $CELL" >&2; exit 2 ;;
esac

case "$SAMPLING" in
  greedy) SAMP=(--temperature 0) ;;
  qwen)   SAMP=(--temperature 0.7 --top-p 0.8 --top-k 20) ;;
  *) echo "unknown sampling: $SAMPLING" >&2; exit 2 ;;
esac
[ "$THINKING" = on ] && THINK=true || THINK=false

OUT=$M0/results${RUN:+/$RUN}/$CONFIG
mkdir -p "$OUT"
NAME="$CELL-c$CONC-$SAMPLING-think$THINKING${RUN:+-s${SEED:-0}}"

# openai-chat + --skip-chat-template: the server applies the template once, and
# enable_thinking reaches it through chat_template_kwargs. No --ignore-eos on real text.
vllm bench serve \
  --backend openai-chat --endpoint /v1/chat/completions --skip-chat-template \
  --model "$MODEL" --port "$PORT" \
  "${DS[@]}" "${SAMP[@]}" \
  --extra-body "{\"chat_template_kwargs\":{\"enable_thinking\":$THINK}}" \
  --num-prompts "$N" --max-concurrency "$CONC" --num-warmups 4 --seed "${SEED:-0}" \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --result-dir "$OUT" --result-filename "$NAME.json" \
  --metadata config="$CONFIG" cell="$CELL" sampling="$SAMPLING" thinking="$THINKING" max_model_len="$MAX_MODEL_LEN" seed="${SEED:-0}" speed_cat="${SPEED_CAT:-}" \
  2>&1 | tee "$OUT/$NAME.log"
