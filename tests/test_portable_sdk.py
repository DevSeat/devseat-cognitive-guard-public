from __future__ import annotations

import pytest

from devseat_cognitive_guard import (
    ContextItem,
    OptimizationProfile,
    PortableContextOptimizer,
    SDK_SCHEMA_VERSION,
)
from devseat_cognitive_guard.bridge import capabilities, process_message
from devseat_cognitive_guard.policy_store import FilePolicyStore
from devseat_cognitive_guard.token_adapters import (
    CallableTokenCounter,
    HuggingFaceTokenizerCounter,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


class DictAdapter:
    def to_items(self, payload):
        return tuple(
            ContextItem(
                key=f"field:{key}",
                content=str(value),
                priority=100 if key == "keep" else 1,
            )
            for key, value in payload.items()
        )

    def from_items(self, original, selected):
        keys = {item.key.split(":", 1)[1] for item in selected}
        return {key: value for key, value in original.items() if key in keys}


def test_portable_sdk_optimizes_items_with_stable_schema() -> None:
    optimizer = PortableContextOptimizer(
        token_counter=CallableTokenCounter(lambda text: len(text.split()))
    )
    result = optimizer.optimize(
        (
            ContextItem("keep", "one two", priority=100),
            ContextItem("drop", "three four five", priority=1),
        ),
        profile=OptimizationProfile(
            token_budget=2,
            selection_strategy="balanced",
        ),
    )

    assert result.schema == SDK_SCHEMA_VERSION
    assert tuple(item.key for item in result.items) == ("keep",)
    assert result.omitted_keys == ("drop",)


def test_portable_sdk_can_optimize_arbitrary_payload_via_adapter() -> None:
    optimizer = PortableContextOptimizer(
        token_counter=CallableTokenCounter(lambda text: len(text.split()))
    )
    result = optimizer.optimize_payload(
        {"keep": "one two", "drop": "three four five"},
        adapter=DictAdapter(),
        profile=OptimizationProfile(token_budget=2),
    )

    assert result.payload == {"keep": "one two"}
    assert result.optimization.omitted_keys == ("field:drop",)


def test_jsonl_bridge_contract_preserves_request_metadata() -> None:
    response = process_message(
        {
            "schema": SDK_SCHEMA_VERSION,
            "request_id": "req-1",
            "product": "test-product",
            "profile": {
                "token_budget": 5,
                "selection_strategy": "balanced",
            },
            "context": [
                {
                    "key": "required",
                    "content": "must keep",
                    "required": True,
                },
                {
                    "key": "noise",
                    "content": "noise " * 100,
                    "priority": 1,
                },
            ],
        }
    )

    assert response["ok"] is True
    assert response["schema"] == SDK_SCHEMA_VERSION
    assert response["request_id"] == "req-1"
    assert response["metadata"]["product"] == "test-product"
    assert [item["key"] for item in response["context"]] == ["required"]


def test_jsonl_bridge_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="unsupported schema"):
        process_message(
            {
                "schema": "devseat.context.v999",
                "context": [],
            }
        )


def test_huggingface_tokenizer_counter_uses_host_tokenizer() -> None:
    counter = HuggingFaceTokenizerCounter(FakeTokenizer())
    assert counter.count("one two three") == 3



def test_bridge_capabilities_handshake() -> None:
    response = process_message(
        {
            "schema": SDK_SCHEMA_VERSION,
            "op": "capabilities",
            "request_id": "hello",
        }
    )

    assert response["ok"] is True
    assert response["request_id"] == "hello"
    assert response["capabilities"]["schema"] == SDK_SCHEMA_VERSION
    assert "balanced" in response["capabilities"]["selection_strategies"]
    assert "chunk_mode" in response["capabilities"]["context_features"]
    assert capabilities()["operations"] == ["capabilities", "optimize"]



def test_bridge_persistent_policy_feedback_round_trip(tmp_path) -> None:
    store = FilePolicyStore(tmp_path)
    context = [
        {
            "key": "critical-001",
            "content": "critical fact",
            "priority": 1,
            "source": "critical",
        },
        {
            "key": "noise-001",
            "content": "noise fact",
            "priority": 10,
            "source": "noise",
        },
    ]

    optimized = process_message(
        {
            "schema": SDK_SCHEMA_VERSION,
            "op": "optimize",
            "request_id": "opt-1",
            "product": "BIO",
            "profile": {
                "token_budget": 2,
                "selection_strategy": "balanced",
            },
            "context": context,
        },
        policy_store=store,
    )
    assert optimized["policy_mode"] == "persistent"
    assert "critical-001" in optimized["omitted_keys"]

    feedback = process_message(
        {
            "schema": SDK_SCHEMA_VERSION,
            "op": "feedback",
            "request_id": "fb-1",
            "product": "BIO",
            "context": context,
            "omitted_keys": optimized["omitted_keys"],
            "raw_quality_pass": True,
            "optimized_quality_pass": False,
            "attributed_keys": ["critical-001"],
        },
        policy_store=store,
    )
    assert feedback["features_touched"] > 0
    assert feedback["policy_summary"]["positive_bias_features"] > 0

    summary = process_message(
        {
            "schema": SDK_SCHEMA_VERSION,
            "op": "policy-summary",
            "product": "BIO",
        },
        policy_store=store,
    )
    assert summary["policy_summary"]["positive_bias_features"] > 0


def test_bridge_capabilities_advertise_feedback_only_with_policy_store(tmp_path) -> None:
    stateless = process_message(
        {
            "schema": SDK_SCHEMA_VERSION,
            "op": "capabilities",
        }
    )
    persistent = process_message(
        {
            "schema": SDK_SCHEMA_VERSION,
            "op": "capabilities",
        },
        policy_store=FilePolicyStore(tmp_path),
    )

    assert "feedback" not in stateless["capabilities"]["operations"]
    assert "feedback" in persistent["capabilities"]["operations"]
    assert persistent["capabilities"]["persistent_policies"] is True
