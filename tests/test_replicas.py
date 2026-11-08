import pytest

from manifold.router.replicas import parse_vllm_metrics

# Shaped like vLLM 0.x /metrics with data parallel size 2: two engine label sets per family.
VLLM_METRICS = """\
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="ibm-granite/granite-4.2-8b"} 3.0
vllm:num_requests_running{engine="1",model_name="ibm-granite/granite-4.2-8b"} 2.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="ibm-granite/granite-4.2-8b"} 4.0
vllm:num_requests_waiting{engine="1",model_name="ibm-granite/granite-4.2-8b"} 1.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="ibm-granite/granite-4.2-8b"} 0.25
vllm:kv_cache_usage_perc{engine="1",model_name="ibm-granite/granite-4.2-8b"} 0.75
# HELP vllm:prefix_cache_queries_total Prefix cache queries, in terms of number of queried tokens.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="ibm-granite/granite-4.2-8b"} 12000.0
vllm:prefix_cache_queries_total{engine="1",model_name="ibm-granite/granite-4.2-8b"} 8000.0
# HELP vllm:prefix_cache_queries_created Prefix cache queries, in terms of number of queried tokens.
# TYPE vllm:prefix_cache_queries_created gauge
vllm:prefix_cache_queries_created{engine="0",model_name="ibm-granite/granite-4.2-8b"} 1.7e+09
# HELP vllm:prefix_cache_hits_total Prefix cache hits, in terms of number of cached tokens.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="ibm-granite/granite-4.2-8b"} 6400.0
vllm:prefix_cache_hits_total{engine="1",model_name="ibm-granite/granite-4.2-8b"} 1600.0
# HELP vllm:request_queue_time_seconds Histogram of time spent in WAITING phase for request.
# TYPE vllm:request_queue_time_seconds histogram
vllm:request_queue_time_seconds_bucket{engine="0",le="0.3",model_name="ibm-granite/granite-4.2-8b"} 7.0
vllm:request_queue_time_seconds_bucket{engine="0",le="+Inf",model_name="ibm-granite/granite-4.2-8b"} 9.0
vllm:request_queue_time_seconds_count{engine="0",model_name="ibm-granite/granite-4.2-8b"} 9.0
vllm:request_queue_time_seconds_sum{engine="0",model_name="ibm-granite/granite-4.2-8b"} 1.5
# HELP python_gc_objects_collected_total Objects collected during gc
# TYPE python_gc_objects_collected_total counter
python_gc_objects_collected_total{generation="0"} 1234.0
"""


def test_parse_sums_counts_and_averages_kv_usage():
    g = parse_vllm_metrics(VLLM_METRICS, ts=42.0)
    assert (g.waiting, g.running) == (5, 5)
    assert g.kv_usage == pytest.approx(0.5)
    assert (g.prefix_hits_total, g.prefix_queries_total) == (8000, 20000)
    assert g.ts == 42.0


def test_parse_missing_families_reads_as_idle():
    g = parse_vllm_metrics("", ts=0.0)
    assert (g.waiting, g.running, g.kv_usage, g.prefix_hits_total) == (0, 0, 0.0, 0)
