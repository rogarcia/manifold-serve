# deploy/runpod — M0: one RTX 5090, one vLLM replica

```bash
# 0. once: runpodctl doctor   (Restricted key, GraphQL Read/Write)
runpodctl gpu list | grep -i 5090
runpodctl pod create --name m0-qwen38 \
  --gpu-id "NVIDIA GeForce RTX 5090" --min-cuda-version 12.8 \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --volume-in-gb 80 --container-disk-in-gb 30 \
  --ports "8000/http,22/tcp" --terminate-after <UTC ISO time>
runpodctl pod get <pod-id>           # until uptimeSeconds > 0
runpodctl ssh info <pod-id>          # prints the ssh command
# on the pod:
git clone https://github.com/rogarcia/manifold-serve /workspace/manifold-serve
bash /workspace/manifold-serve/deploy/runpod/bootstrap.sh            # once: uv + vLLM on /workspace
bash /workspace/manifold-serve/deploy/runpod/serve_qwen38_27b_nvfp4.sh 2>&1 | tee /workspace/vllm-startup.log
# second ssh session, on the pod:
cd /workspace/manifold-serve && uv sync
uv run python -m manifold.loadgen --base-url http://localhost:8000 --model qwen3.8-27b-nvfp4 --rate 2 --duration 120
runpodctl pod delete <pod-id>        # stop the meter. `stop` keeps the volume billed.
```

Lesson learned the expensive way: `--docker-args` is appended to the image ENTRYPOINT, not a
replacement command. `vllm/vllm-openai` has entrypoint `vllm` and no sshd, so "sleep infinity"
became `vllm sleep infinity` in a crash loop with no way to log in.
