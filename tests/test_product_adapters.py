from __future__ import annotations

from devseat_cognitive_guard import (
    OptimizationProfile,
    PortableContextOptimizer,
)
from devseat_cognitive_guard.integrations.message_history import (
    MessageHistoryAdapter,
)
from devseat_cognitive_guard.token_adapters import CallableTokenCounter


def _optimizer():
    return PortableContextOptimizer(
        token_counter=CallableTokenCounter(lambda text: len(text.split()))
    )


def test_message_history_keeps_system_and_recent_turns() -> None:
    messages = [
        {"role": "system", "content": "system rules"},
        {"role": "user", "content": "very old user context context context"},
        {"role": "assistant", "content": "old answer answer answer"},
        {"role": "user", "content": "recent question"},
        {"role": "assistant", "content": "recent answer"},
    ]
    adapter = MessageHistoryAdapter(keep_last=2)
    result = _optimizer().optimize_payload(
        messages,
        adapter=adapter,
        profile=OptimizationProfile(token_budget=6),
    )

    assert result.payload[0] is messages[0]
    assert result.payload[-2:] == messages[-2:]
    assert messages[1] not in result.payload
