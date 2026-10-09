from __future__ import annotations

from devseat_cognitive_guard.integrations.bio_laya import (
    BioLayaContextOptimizer,
    OptimizingLayaRouter,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class FakeAgent:
    def __init__(self):
        self.tok = FakeTokenizer()
        self.last_state = None

    def system_one(self, state, questions):
        self.last_state = state
        return {
            "model": "fake-laya",
            "answers": {"ok": True},
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }


class FakeRouter:
    def __init__(self):
        self.agent = FakeAgent()

    def route(self, state, questions, model=None, task=None, lang=None):
        return {
            "model": "english",
            "repo": "fake",
            "reason": "test",
        }

    def load(self, name):
        return self.agent


def test_bio_dict_adapter_preserves_original_value_types() -> None:
    agent = FakeAgent()
    adapter = BioLayaContextOptimizer.from_laya_agent(
        agent,
        required_fields=("current",),
        field_priorities={"history": 1},
    )
    state = {
        "current": {"value": 7},
        "history": [1, 2, 3, 4, 5],
    }

    result = adapter.optimize_state(state, token_budget=1)

    assert result.state == {"current": {"value": 7}}
    assert isinstance(result.state["current"], dict)


def test_bio_list_adapter_keeps_recent_turns_required() -> None:
    agent = FakeAgent()
    adapter = BioLayaContextOptimizer.from_laya_agent(
        agent,
        keep_last_turns=2,
    )
    state = [
        {"role": "user", "content": "old old old old"},
        {"role": "assistant", "content": "recent one"},
        {"role": "user", "content": "recent two"},
    ]

    result = adapter.optimize_state(state, token_budget=4)

    assert result.state[-2:] == state[-2:]


def test_optimizing_laya_router_wraps_existing_router() -> None:
    router = FakeRouter()
    wrapped = OptimizingLayaRouter(
        router,
        token_budget=4,
        required_fields=("current",),
        field_priorities={"noise": 1},
    )
    state = {
        "current": "answer",
        "noise": "many old tokens that are not needed",
    }

    result = wrapped.predict(
        state,
        {"q": {"type": "noul", "instructions": "test"}},
    )

    assert router.agent.last_state == {"current": "answer"}
    assert result["routing"]["model"] == "english"
    assert result["context_optimization"]["saved_context_tokens"] > 0
