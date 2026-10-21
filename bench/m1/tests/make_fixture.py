"""Write the summarize.py self-test fixture. Usage: python3 tests/make_fixture.py tests/fixture

summarize.py --self-test asserts numbers worked out by hand from the values below.
"""
import json, sys
from pathlib import Path
root = Path(sys.argv[1])

def rec(conv, start, turns, ttft, tpot, lat=1000.0, client=0):
    return dict(ttft_ms=ttft, tpot_ms=tpot, latency_ms=lat, start_time_ms=start, input_num_turns=turns,
                input_num_tokens=100*turns, output_num_tokens=50, output_num_chunks=49,
                output_num_first_chunk_tokens=1, approx_cached_percent=0.0, conversation_id=conv, client_id=client)

def dec(sess, idx, t, spilled=False, status=200, error=None):
    url = ["http://localhost:8001", "http://localhost:8002"][idx]
    return dict(request_id=f"r{t}", session_id=sess, policy="x", chosen=url, replica_index=idx, spilled=spilled,
                prompt_tokens=100, total_blocks=2, matched_blocks={}, scores={}, t_received=t, t_sent=t,
                t_first_byte=t, t_done=t, status=status, error=error)

def prom(d):
    return "".join(f'{k}{{model_name="m"}} {v}\n' for k, v in d.items())

def write_cell(d, meta, stats, decisions, before, after, samples):
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    (d / "stats.json").write_text(json.dumps(stats, indent=1))
    (d / "decisions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in decisions))
    for r in "ab":
        (d / f"metrics_before_{r}.txt").write_text(prom(before[r]))
        (d / f"metrics_after_{r}.txt").write_text(prom(after[r]))
    (d / "samples.tsv").write_text("t\treplica\twaiting\trunning\tkv_usage\n" + "".join(
        f"{t}\t{r}\t{w}\t1\t0.5\n" for t, r, w in samples))
    (d / "DONE").touch()

H, Q = "vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"
QS, QC = "vllm:request_queue_time_seconds_sum", "vllm:request_queue_time_seconds_count"
PS, PC = "vllm:request_prefill_time_seconds_sum", "vllm:request_prefill_time_seconds_count"
before = {"a": {H: 100, Q: 1000, QS: 1.0, QC: 10, PS: 0, PC: 0}, "b": {H: 0, Q: 0, QS: 0, QC: 0, PS: 0, PC: 0}}
after = {"a": {H: 600, Q: 2000, QS: 2.0, QC: 20, PS: 0.4, PC: 10}, "b": {H: 100, Q: 1000, QS: 0.5, QC: 10, PS: 0.2, PC: 10}}
samples = [(1000 + t, r, 0) for t in range(0, 8) for r in "ab"]

def meta(policy, clients, seed, pressure="pressure", load="70", run="fixture"):
    return dict(run=run, model_tag="q35-9b", pressure=pressure, workload="sharegpt", policy=policy,
                clients=clients, seed=seed, nreq=6, load_label=load, active_requested=64, active_effective=64,
                t_start=1000, t_end=1010)

for policy, seed, scale in (("prefix_aware", 1, 1.0), ("prefix_aware", 2, 1.1), ("least_load", 1, 1.3)):
    kept = [v * scale for v in (100, 200, 300, 400)]
    stats = [rec("x", 1000, 1, 5000, 99), rec("y", 1500, 1, 5000, 99),
             rec("x", 3000, 3, kept[0], 10), rec("y", 3500, 3, kept[1], 20),
             rec("x", 5000, 5, kept[2], 30), rec("y", 5500, 5, kept[3], 40)]
    decisions = [dec("x", 0, 1.0), dec("y", 1, 1.5), dec("z", 1, 1.7, status=500, error="upstream"),
                 dec("x", 0, 3.0), dec("y", 1, 3.5), dec("x", 0, 5.0), dec("y", 0, 5.5, spilled=True)]
    write_cell(root / "matrix/fixture/q35-9b/pressure/sharegpt" / f"{policy}-c2-s{seed}",
               meta(policy, 2, seed), stats, decisions, before, after, samples)

for clients, rps, ttft in ((2, 1, 200), (4, 2, 400), (8, 4, 1500), (16, 8, 3000)):
    k = rps * 10
    stats = [rec("w", 0, 1, 9999, 1)] + [
        rec(f"c{i}", 1000 + i * 9000 / (k - 1), 1, ttft, 10) for i in range(k)]
    decisions = [dec("w", 0, 0.0)] + [dec(f"c{i}", i % 2, 1.0 + i) for i in range(k)]
    write_cell(root / "calib/calib/q35-9b/nocap/sharegpt" / f"least_load-c{clients}-s0",
               meta("least_load", clients, 0, pressure="nocap", load="calib", run="calib"),
               stats, decisions, before, after,
               [(1000 + t, r, t if clients == 4 else 0) for t in range(0, 8) for r in "ab"])
