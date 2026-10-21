"""Run the vendored ShareGPT converter with two corrections, leaving the vendored file unchanged.

Usage: python prep_sharegpt.py INPUT OUTPUT [converter flags...]
Positionals come first. The flags are those of vendor/multi_turn/convert_sharegpt_to_openai.py.
Run with the bench venv's python (it needs pandas, tqdm and transformers).

1. The converter's language filter is inverted (see vendor/VENDOR.md). This restores the
   intended predicate: the length bounds, and ASCII-only content.
2. The converter keeps conversations that end on a user message. The multi-turn bench asserts
   outside its try block when it reaches such a turn, the client process dies without sending
   its termination signal, and the benchmark waits forever. Each conversation is trimmed to an
   even number of messages so it ends on an assistant reply.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor" / "multi_turn"))

import convert_sharegpt_to_openai as conv  # noqa: E402


def content_is_valid(content: str, min_content_len: int | None, max_content_len: int | None) -> bool:
    if min_content_len and len(content) < min_content_len:
        return False
    if max_content_len and len(content) > max_content_len:
        return False
    return content.isascii()


def trim_to_assistant_end(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    trimmed = 0
    for item in data:
        msgs = item["messages"]
        if len(msgs) % 2:
            del msgs[-1]
            trimmed += 1
    data = [item for item in data if len(item["messages"]) >= 2]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"trimmed a trailing user message from {trimmed:,} conversations; {len(data):,} remain")


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1].startswith("-") or sys.argv[2].startswith("-"):
        sys.exit(__doc__)
    conv.content_is_valid = content_is_valid
    conv.main()
    trim_to_assistant_end(Path(sys.argv[2]))
