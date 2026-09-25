# Stream a few chat requests and log every SSE chunk with its arrival time, to see what the
# first chunks contain and where `vllm bench serve` starts its TTFT clock.
# Usage (on the pod, server running): python probe-ttft.py OUT.jsonl [N_PROMPTS]
# Runs with the pod venv's python (httpx and datasets come with vllm[bench]).
import json, os, sys, time

import httpx
from datasets import load_dataset

out_path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
url = f"http://localhost:{os.environ.get('PORT', '8000')}/v1/chat/completions"
config = open(os.path.join(os.environ["M0"], "current-config")).read().strip()

ds = load_dataset("philschmid/mt-bench", split="train")
prompts = [ds[i]["turns"][0] for i in range(n)]

with httpx.Client(timeout=300) as client, open(out_path, "a") as f:
    for i, prompt in enumerate(prompts):
        body = {
            "model": os.environ["MODEL"],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.perf_counter()
        chunks = []
        with client.stream("POST", url, json=body) as r:
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                t = (time.perf_counter() - t0) * 1000
                d = json.loads(line[6:])
                delta = d["choices"][0].get("delta", {}) if d.get("choices") else {}
                chunks.append({"ms": round(t, 1), "content": delta.get("content"),
                               "reasoning": delta.get("reasoning") or delta.get("reasoning_content"),
                               "role": delta.get("role"), "usage": d.get("usage")})
        first_any = chunks[0]["ms"]
        first_content = next((c["ms"] for c in chunks if c["content"]), None)
        rec = {"config": config, "prompt": i, "first_chunk_ms": first_any,
               "first_content_ms": first_content, "chunks": chunks[:12]}
        f.write(json.dumps(rec) + "\n")
        print(f"{config} p{i}: first chunk {first_any} ms, first content {first_content} ms")
        for c in chunks[:6]:
            print("   ", c["ms"], repr(c["content"]), repr(c["reasoning"]), c["role"] or "")
