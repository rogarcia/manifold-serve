# What continuous batching actually does to your latency curves

_Draft template for post #1 (M0). Replace every placeholder with measured data before publishing._

## Setup

- Hardware: [RTX 5090 32GB, RunPod secure cloud, region]
- Model: Inferact/Qwen3.8-27B-NVFP4 (27B dense, NVFP4 weights, FP8 KV), vLLM [version], `--enable-prefix-caching --enforce-eager`
- Load: open-loop Poisson arrivals, multi-turn chat sessions with shared prefixes
  (generator: [link to loadgen])

## The experiment

Sweep arrival rate from [0.5] to [8] req/s and record TTFT and end-to-end latency
percentiles at each step.

[CHART: TTFT p50/p99 vs arrival rate]
[CHART: throughput (output tok/s) vs arrival rate]

## What the curves say

1. [Where does p99 detach from p50, and why: queueing at the scheduler]
2. [The knee: at what rate the GPU saturates and what TTFT does after that]
3. [Prefix caching effect visible between turn 1 and turn N of a session]

## Why this matters at fleet scale

One replica's saturation knee is the input to every interesting fleet question: routing
policy (send the request where the knee is furthest away), autoscaling (add a replica
before the knee, not after), and deployment safety (a canary past its knee looks broken
even when the model is fine). Next post: routing policies across two replicas.

---

_This is post 1 of a build-in-public series on inference serving infrastructure.
Code and raw CSVs: [repo link]._
