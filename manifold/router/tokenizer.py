"""Prompt tokenization for the prefix index: the rendered chat template as token ids."""

from __future__ import annotations

import json
from typing import Any, Protocol


class Tokenizer(Protocol):
    def encode_chat(self, messages: list[dict], template_kwargs: dict[str, Any]) -> list[int]: ...

    def encode_text(self, text: str) -> list[int]: ...


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else json.dumps(content, sort_keys=True)


def render_chat(messages: list[dict]) -> str:
    """A fixed stand-in for a chat template: the same messages always render the same text."""
    parts = [f"<|{m.get('role', '')}|>\n{_content_text(m.get('content'))}\n" for m in messages]
    return "".join(parts) + "<|assistant|>\n"


class BytesTokenizer:
    """UTF-8 bytes as token ids. For tests and the sim; ignores template kwargs."""

    def encode_chat(self, messages: list[dict], template_kwargs: dict[str, Any]) -> list[int]:
        return list(render_chat(messages).encode())

    def encode_text(self, text: str) -> list[int]:
        return list(text.encode())


class HFTokenizer:
    def __init__(self, repo_id: str):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(repo_id)

    def encode_chat(self, messages: list[dict], template_kwargs: dict[str, Any]) -> list[int]:
        return list(
            self._tok.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                **template_kwargs,
            )
        )

    def encode_text(self, text: str) -> list[int]:
        return list(self._tok.encode(text, add_special_tokens=False))


def make_tokenizer(spec: str) -> Tokenizer:
    return BytesTokenizer() if spec == "bytes" else HFTokenizer(spec)
