from __future__ import annotations

import math
from typing import Protocol


class TokenCounter(Protocol):
    """Minimal tokenizer abstraction used by the context optimizer."""

    def count(self, text: str) -> int:
        ...


class ApproxTokenCounter:
    """Dependency-free conservative token estimate.

    ASCII text is estimated at roughly four characters per token. Non-ASCII
    code points are counted individually, which is intentionally conservative
    for Korean/CJK-heavy context. Callers that need exact accounting should
    inject a provider/model tokenizer implementing TokenCounter.
    """

    def count(self, text: str) -> int:
        if not text:
            return 0

        ascii_chars = 0
        non_ascii_chars = 0
        for char in text:
            if ord(char) < 128:
                ascii_chars += 1
            else:
                non_ascii_chars += 1

        ascii_tokens = math.ceil(ascii_chars / 4) if ascii_chars else 0
        return ascii_tokens + non_ascii_chars
