"""Poll each replica's /metrics once a second and append gauges to a TSV until killed.

Usage: python sampler.py OUT.tsv a=http://localhost:8001 b=http://localhost:8002
Columns: t (Unix seconds), replica, waiting, running, kv_usage. Stdlib only.
"""

import sys
import time
import urllib.request

GAUGES = {
    "vllm:num_requests_waiting": "waiting",
    "vllm:num_requests_running": "running",
    "vllm:kv_cache_usage_perc": "kv_usage",
}


def read_gauges(url: str) -> dict[str, float]:
    with urllib.request.urlopen(url + "/metrics", timeout=2) as r:
        text = r.read().decode()
    out = dict.fromkeys(GAUGES.values(), 0.0)
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name in GAUGES:
            out[GAUGES[name]] += float(line.rsplit(" ", 1)[1])
    return out


def main() -> None:
    out_path, targets = sys.argv[1], [arg.split("=", 1) for arg in sys.argv[2:]]
    with open(out_path, "a", buffering=1) as f:
        if f.tell() == 0:
            f.write("t\treplica\twaiting\trunning\tkv_usage\n")
        while True:
            tick = time.time()
            for name, url in targets:
                try:
                    g = read_gauges(url)
                except OSError:
                    continue
                f.write(f"{tick:.3f}\t{name}\t{g['waiting']:.0f}\t{g['running']:.0f}\t{g['kv_usage']:.4f}\n")
            time.sleep(max(0.0, 1.0 - (time.time() - tick)))


if __name__ == "__main__":
    main()
