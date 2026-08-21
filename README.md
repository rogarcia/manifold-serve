# manifold-serve

An open inference serving platform: cache-aware request routing, workload-class autoscaling, and
canary model deployments for LLM fleets on Kubernetes.

> Serving LLMs is a distributed systems problem. Once you have more than one replica, the
> interesting questions stop being about the model and start being about the fleet: where does
> this request go, how many replicas should exist right now, and how do you ship a new model
> without breaking the ones in flight? `manifold-serve` is a working, measured exploration of those
> questions on real hardware, at a budget an individual can afford.

Built in public. Every design decision ships with a benchmark and a writeup: see `[docs/posts/](docs/posts/)`.

## Architecture

```
                        +---------------------------+
   clients  ──────────► │  router (data plane)      │
   (OpenAI-compatible)  │  policies: round-robin /  │
                        │  least-outstanding /      │
                        │  prefix-cache-aware       │
                        +------------+--------------+
                                     │
                 ┌───────────────────┼───────────────────┐
                 ▼                   ▼                   ▼
          +-------------+    +-------------+     +-------------+
          | vLLM replica|    | vLLM replica|     | vLLM replica|
          |  (GPU 0)    |    |  (GPU 1)    |     |  (spot)     |
          +-------------+    +-------------+     +-------------+
                 ▲                   ▲                   ▲
                 └────────── autoscaler (control plane) ─┘
                    scales on queue depth, tokens in flight,
                    KV-cache utilization; workload classes with
                    priority + preemption; scale-to-zero on spot
```



## Roadmap

- [ ] **M0**: Serve a real model (vLLM), async load generator, first published TTFT/TPOT/throughput curves
- [ ] **M1**: Router v1 with pluggable policies; measured comparison: round-robin vs least-outstanding vs prefix-cache-aware
- [ ] **M2**: Custom-metrics autoscaler on Kubernetes; interactive vs batch workload classes; scale-to-zero on spot GPUs
- [ ] **M3**: Canary model deployments gated by automated evals, auto-promote/rollback
- [ ] **M4**: Rust port of the router data plane (axum/tokio), side-by-side latency comparison
- [ ] **M5**: Two-region latency-aware routing + failover; fleet-scale discrete-event simulator



## Quickstart

Requires one GPU host (a rented RTX 5090 or PRO 6000 works) running two vLLM replicas, or any two OpenAI-compatible endpoints.

```bash
# 1. Start two vLLM replicas sharing one GPU (see deploy/docker-compose.yml)
docker compose -f deploy/docker-compose.yml up -d

# 2. Run the router
pip install -e .
MANIFOLD_BACKENDS="http://localhost:8001,http://localhost:8002" \
MANIFOLD_POLICY=least_outstanding \
python -m manifold.router

# 3. Drive load and measure
python -m manifold.loadgen --base-url http://localhost:8000 \
  --rate 4 --duration 120 --sessions 32 --out bench/results.csv
```

The load generator uses open-loop Poisson arrivals and multi-turn chat sessions with shared
prefixes, so prefix-cache effects are actually exercised.

## Benchmarks

Coming with M0. Every number published here is reproducible from `bench/`.


| policy    | model | replicas | req/s | TTFT p50 | TTFT p99 | output tok/s |
| --------- | ----- | -------- | ----- | -------- | -------- | ------------ |
| *pending* |       |          |       |          |          |              |




## Why "manifold"

A manifold takes one intake and distributes flow across many outlets without losing
pressure. Same job here, with tokens.

## License

MIT