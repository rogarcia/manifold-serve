"""HTTP surface of the fake engine, in vLLM's OpenAI-compatible shapes."""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST

from manifold.router.tokenizer import BytesTokenizer

from .engine import Engine, SimConfig

log = logging.getLogger("manifold.sim")
DEFAULT_MAX_TOKENS = 64


def create_app(config: SimConfig) -> FastAPI:
    engine = Engine(config)
    tokenizer = BytesTokenizer()
    kv_tokens = config.kv_blocks * config.block_tokens

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info("Setting attention block size to %d tokens", config.block_tokens)
        log.info("GPU KV cache size: %s tokens", f"{kv_tokens:,}")
        yield

    app = FastAPI(title="manifold-sim", lifespan=lifespan)
    app.state.engine = engine

    @app.get("/health")
    async def health() -> Response:
        return Response(status_code=200)

    @app.get("/version")
    async def version() -> dict:
        return {"version": "sim"}

    @app.get("/v1/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": config.model, "object": "model", "created": 0, "owned_by": "vllm",
                      "root": config.model, "max_model_len": kv_tokens}],
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(engine.render_metrics(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/reset_prefix_cache")
    async def reset_prefix_cache() -> Response:
        engine.reset_prefix_cache()
        return Response(status_code=200)

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Response:
        body = await request.json()
        messages = body.get("messages")
        if not isinstance(messages, list):
            return JSONResponse({"error": {"message": "messages must be a list"}}, 400)
        tokens = tokenizer.encode_chat(messages, {})
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens")
        max_tokens = max_tokens if isinstance(max_tokens, int) else DEFAULT_MAX_TOKENS
        rid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        usage_opt = (body.get("stream_options") or {}).get("include_usage", False)

        def chunk(delta: dict, finish: str | None) -> str:
            payload = {"id": rid, "object": "chat.completion.chunk", "created": created,
                       "model": config.model,
                       "choices": [{"index": 0, "delta": delta, "logprobs": None,
                                    "finish_reason": finish}]}
            return f"data: {json.dumps(payload)}\n\n"

        def usage(completion: int) -> dict:
            return {"prompt_tokens": len(tokens), "completion_tokens": completion,
                    "total_tokens": len(tokens) + completion}

        if not body.get("stream"):
            async with aclosing(engine.generate(tokens, max_tokens)) as gen:
                parts = []
                async for text in gen:
                    parts.append(text)
                    if await request.is_disconnected():
                        return Response(status_code=499)
            return JSONResponse({
                "id": rid, "object": "chat.completion", "created": created,
                "model": config.model,
                "choices": [{"index": 0, "message": {"role": "assistant",
                                                     "content": "".join(parts)},
                             "logprobs": None, "finish_reason": "length"}],
                "usage": usage(len(parts)),
            })

        async def sse() -> AsyncIterator[str]:
            n = 0
            async with aclosing(engine.generate(tokens, max_tokens)) as gen:
                async for text in gen:
                    if n == 0:
                        yield chunk({"role": "assistant", "content": ""}, None)
                    n += 1
                    yield chunk({"content": text}, None)
            yield chunk({}, "length")
            if usage_opt:
                payload = {"id": rid, "object": "chat.completion.chunk", "created": created,
                           "model": config.model, "choices": [], "usage": usage(n)}
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return app


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m manifold.sim", description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--model", default=SimConfig.model)
    p.add_argument("--block-tokens", type=int, default=SimConfig.block_tokens)
    p.add_argument("--kv-blocks", type=int, default=SimConfig.kv_blocks)
    p.add_argument("--max-num-seqs", type=int, default=SimConfig.max_num_seqs)
    p.add_argument("--prefill-us-per-token", type=float, default=SimConfig.prefill_us_per_token)
    p.add_argument("--decode-ms-per-token", type=float, default=SimConfig.decode_ms_per_token)
    p.add_argument("--decode-ms-per-running", type=float,
                   default=SimConfig.decode_ms_per_running)
    p.add_argument("--seed", type=int, default=SimConfig.seed)
    p.add_argument("--start-delay-s", type=float, default=0.0,
                   help="sleep before binding the port, as an engine does while it loads")
    p.add_argument("--shutdown-timeout-s", type=int, default=None,
                   help="after SIGTERM, wait this long for running requests")
    a = p.parse_args()
    if min(a.block_tokens, a.kv_blocks, a.max_num_seqs) < 1:
        p.error("--block-tokens, --kv-blocks and --max-num-seqs must be positive")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s [sim] %(message)s",
                        datefmt="%m-%d %H:%M:%S")
    config = SimConfig(model=a.model, block_tokens=a.block_tokens, kv_blocks=a.kv_blocks,
                       max_num_seqs=a.max_num_seqs, prefill_us_per_token=a.prefill_us_per_token,
                       decode_ms_per_token=a.decode_ms_per_token,
                       decode_ms_per_running=a.decode_ms_per_running, seed=a.seed)
    import uvicorn

    time.sleep(a.start_delay_s)
    uvicorn.run(create_app(config), host=a.host, port=a.port, log_level="warning",
                timeout_graceful_shutdown=a.shutdown_timeout_s)


if __name__ == "__main__":
    main()
