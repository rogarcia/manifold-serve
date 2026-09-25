# TTFT test: why is chat/math TTFT lower under MTP than on baseline, but code TTFT higher?
# Sends sequential streaming requests (10 mt-bench + 10 InstructCoder prompts) at max_tokens 1
# and 16. Each prompt starts with a unique nonce so nothing hits the prefix cache. Records
# client-side time to first chunk and, per group, the deltas of the server's own
# TTFT / prefill / queue histograms from /metrics.
# Usage (on the pod, server running, no other traffic): python ttft-test.py OUT.jsonl
import json, os, re, sys, time, uuid

import httpx
from datasets import load_dataset

out_path = sys.argv[1]
base = f"http://localhost:{os.environ.get('PORT', '8000')}"
config = open(os.path.join(os.environ["M0"], "current-config")).read().strip()

chat = [r["turns"][0] for r in load_dataset("philschmid/mt-bench", split="train").select(range(10))]
code = [f"{r['input']}\n\n{r['instruction']} Just output the code, do not include any explanation."
        for r in load_dataset("likaixin/InstructCoder", split="train").select(range(10))]

METRICS = ["time_to_first_token_seconds", "request_prefill_time_seconds", "request_queue_time_seconds"]


def scrape(client):
    text = client.get(f"{base}/metrics").text
    out = {}
    for m in METRICS:
        for kind in ("sum", "count"):
            hit = re.search(rf"^vllm:{m}_{kind}\{{[^}}]*\}} (\S+)$", text, re.M)
            out[f"{m}_{kind}"] = float(hit.group(1)) if hit else float("nan")
    return out


def one(client, prompt, max_tokens):
    body = {"model": os.environ["MODEL"],
            "messages": [{"role": "user", "content": f"[{uuid.uuid4().hex}]\n{prompt}"}],
            "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter(); first = None; usage = None
    with client.stream("POST", f"{base}/v1/chat/completions", json=body) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("choices") and first is None:
                first = (time.perf_counter() - t0) * 1000
            usage = d.get("usage") or usage
    total = (time.perf_counter() - t0) * 1000
    return first, total, usage


with httpx.Client(timeout=300) as client, open(out_path, "a") as f:
    for p in chat[:3]:
        one(client, p, 4)  # warmup, discarded
    for max_tokens in (1, 16):
        for name, prompts in (("chat", chat), ("code", code)):
            before = scrape(client)
            rows = [one(client, p, max_tokens) for p in prompts]
            after = scrape(client)
            d = {k: after[k] - before[k] for k in after}
            n = d["time_to_first_token_seconds_count"] or float("nan")
            rec = {"config": config, "set": name, "max_tokens": max_tokens,
                   "client_ttft_ms": [round(r[0], 1) for r in rows],
                   "client_total_ms": [round(r[1], 1) for r in rows],
                   "prompt_tokens": [r[2]["prompt_tokens"] for r in rows],
                   "completion_tokens": [r[2]["completion_tokens"] for r in rows],
                   "server_ttft_mean_ms": round(1000 * d["time_to_first_token_seconds_sum"] / n, 1),
                   "server_prefill_mean_ms": round(1000 * d["request_prefill_time_seconds_sum"] / n, 1),
                   "server_queue_mean_ms": round(1000 * d["request_queue_time_seconds_sum"] / n, 2)}
            f.write(json.dumps(rec) + "\n")
            ct = sorted(rec["client_ttft_ms"])
            print(f"{config:14s} {name} max_tokens={max_tokens:2d}: client TTFT p50 {ct[len(ct)//2]:6.1f} ms | "
                  f"server TTFT {rec['server_ttft_mean_ms']} prefill {rec['server_prefill_mean_ms']} "
                  f"queue {rec['server_queue_mean_ms']} | prompt tok ~{sum(rec['prompt_tokens'])//len(rows)}")
