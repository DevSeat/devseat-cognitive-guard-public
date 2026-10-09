from __future__ import annotations

from typing import Any


class HuggingFaceTokenizerCounter:
    """Use an already-loaded Hugging Face style tokenizer as TokenCounter."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def count(self, text: str) -> int:
        encoded = self.tokenizer(text, add_special_tokens=False)
        ids = encoded["input_ids"]
        return len(ids)


class CallableTokenCounter:
    """Wrap any callable that maps text to an integer token count."""

    def __init__(self, counter) -> None:
        self.counter = counter

    def count(self, text: str) -> int:
        return int(self.counter(text))
