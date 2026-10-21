# Vendored code

## vLLM multi-turn benchmark

`multi_turn/` is an unmodified copy of `benchmarks/multi_turn/` from vLLM at tag `v0.31.0`
(commit `db9527a46873454610df6dbedf79a36d6bf1a7f6`), downloaded on 2026-10-16 from
`https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/benchmarks/multi_turn/<file>`.

`../fetch-vendor.sh` downloads the files again and checks them against the table below. The
table is the source of truth for the hashes; the script reads it.

| File | sha256 |
|---|---|
| benchmark_serving_multi_turn.py | e508f77895eddb1c45d4b03fd632a7a1774f0da692215e15b07280913d3b846b |
| bench_dataset.py | 75d0eaae586a3b98661392a99f7e8172892ed0fb5a862d3f0c2dc6c05e0b0fa5 |
| bench_utils.py | b968e44ef6260fd1cea8410b3bf4cab751d9d4ad14bdc43f6268d113ad81a548 |
| convert_sharegpt_to_openai.py | fcd361df5eeae1166640c1b7e47f202125a18d47066b3e8de05dbfefd10c83c2 |
| README.md | a7d0fa406d76d0f6c6d19ff5823c096cf7972bffdfe21f60440b284fe077c22c |
| generate_multi_turn.json | 01354996dff57e1b1638c63d66cd07b299af6b932dd7e7f7d23bf2d72795b1ee |
| requirements.txt | ca7dd4710b6eeb69a13bcff6475bd6fa39eb50a48986f4135c8c82bcac04eb1e |

### License

vLLM is licensed under the Apache License, Version 2.0
(`https://github.com/vllm-project/vllm/blob/v0.31.0/LICENSE`). The Python files carry the
upstream header `SPDX-License-Identifier: Apache-2.0` and
`SPDX-FileCopyrightText: Copyright contributors to the vLLM project`. The other three files
have no header and are covered by the same license. Nothing in `multi_turn/` was changed.

### Known upstream defects, and how M1 handles them

1. `convert_sharegpt_to_openai.py` is meant to drop conversations with non-English text, but
   `content_is_valid()` returns `has_non_english_chars(content)`, which is
   `not content.isascii()`. The filter is inverted: it keeps only conversations in which every
   message contains a non-ASCII character. Present at `v0.31.0` and on `main` as of 2026-10-16.
2. The converter keeps conversations that end on a user message (508 of 4,000 in the M1
   sample). `benchmark_serving_multi_turn.py` reaches that turn with
   `assert current_turn < len(messages)` outside its `try` block, so the client process dies
   without putting its termination signal on the queue, and `main_mp()` waits for it forever.
   Found by reading the code, not observed.
3. At the end of `main_mp()` the benchmark drains its task queue with
   `while not task_queue.empty(): task_queue.get()` and then calls `task_queue.join_thread()`.
   `empty()` can return True while the queue's feeder thread still holds conversations, and
   `join_thread()` then waits forever, before `--stats-json-output` is written. Any run with
   `-n` leaves conversations unused. Reproduced on 2026-10-16 with a 40-line script that uses
   the same queue pattern (4,000 items of 3,000 characters, 20 consumed), on macOS with both
   the fork and spawn start methods; the first laptop dry run hung the same way. Not yet seen
   on Linux, but the mechanism (a pipe plus a feeder thread) is the same there.
4. `benchmark_serving_multi_turn.py` imports `redact_sensitive_namespace` from
   `vllm.benchmarks.lib.utils`, and `requirements.txt` does not list vLLM. The pod has vLLM.
   The laptop dry run puts `../tests/vllm_shim/` on `PYTHONPATH` for the benchmark only.

`../prep_sharegpt.py` imports the vendored converter, replaces `content_is_valid()` with the
intended predicate (the length bounds, and `content.isascii()`), runs the module's own `main()`,
and then trims each conversation to an even number of messages. `../run_bench.py` imports the
vendored benchmark and runs its `main()` with `multiprocessing.Queue.join_thread()` told not to
wait in the main process, where the leftover queue data is discarded anyway. The vendored files
stay byte-identical, so the hash check still passes.
