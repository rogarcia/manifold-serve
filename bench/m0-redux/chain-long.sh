#!/usr/bin/env bash
source /workspace/m0-redux/scripts/env.sh
until grep -q PREP_END $M0/logs/speed-bench-prepare.log; do sleep 10; done
cd $M0/data && python speed-bench-prepare.py --config throughput_1k --output_dir $M0/data/speed_bench >> $M0/logs/speed-bench-prepare.log 2>&1 || echo "PREP_FAILED throughput_1k"
ls -la $M0/data/speed_bench
for b in 1k 8k 16k 32k; do [ -s $M0/data/speed_bench/throughput_$b.jsonl ] || { echo "MISSING throughput_$b, not starting run-long"; date -u > $M0/DONE; exit 1; }; done
/workspace/m0-redux/scripts/run-long.sh
