from devseat_cognitive_guard import (
    ApproxTokenCounter,
    ContextItem,
    ContextOptimizer,
    ContextRequest,
)
from devseat_cognitive_guard.context_cli import _parse_request, _serialize


class WordTokenCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def _context():
    return (
        ContextItem(
            "rules",
            "Do not change public APIs.",
            stable=True,
            required=True,
            priority=100,
            source="repo",
            revision="1",
        ),
        ContextItem(
            "duplicate",
            "Do not change public APIs.",
            stable=True,
            priority=10,
        ),
        ContextItem("error", "ValueError on empty input.", priority=90),
        ContextItem("noise", "N" * 3000, priority=1),
    )


def test_reduces_duplicate_and_low_priority_context() -> None:
    result = ContextOptimizer().optimize(
        ContextRequest(context=_context(), context_budget_chars=1200)
    )
    assert "duplicate" in result.reduced_context.duplicate_keys
    assert "noise" in result.reduced_context.omitted_keys
    assert result.reduction_ratio > 0
    assert result.saved_context_tokens > 0


def test_required_context_survives_budget() -> None:
    result = ContextOptimizer().optimize(
        ContextRequest(
            context=(ContextItem("required", "R" * 3000, required=True),),
            context_budget_chars=1000,
        )
    )
    assert result.reduced_context.items[0].key == "required"


def test_duplicate_prefers_required_representative() -> None:
    result = ContextOptimizer().optimize(
        ContextRequest(
            context=(
                ContextItem("optional", "Same.", priority=100),
                ContextItem("required", "Same.", required=True, priority=1),
            )
        )
    )
    assert tuple(item.key for item in result.reduced_context.items) == ("required",)
    assert result.reduced_context.duplicate_keys == ("optional",)


def test_delta_omits_unchanged_stable_context() -> None:
    optimizer = ContextOptimizer()
    first = optimizer.optimize(ContextRequest(context=_context()))
    second = optimizer.optimize(
        ContextRequest(
            context=_context(),
            delta_context=True,
            known_context_fingerprints=first.context_fingerprints,
        )
    )
    assert "rules" in second.reduced_context.unchanged_keys
    assert all(item.key != "rules" for item in second.reduced_context.items)
    assert second.saved_context_tokens > 0


def test_revision_change_invalidates_delta() -> None:
    optimizer = ContextOptimizer()
    first = optimizer.optimize(
        ContextRequest(
            context=(ContextItem("rules", "Rule A", stable=True, revision="1"),)
        )
    )
    second = optimizer.optimize(
        ContextRequest(
            context=(ContextItem("rules", "Rule A", stable=True, revision="2"),),
            delta_context=True,
            known_context_fingerprints=first.context_fingerprints,
        )
    )
    assert second.reduced_context.unchanged_keys == ()
    assert second.reduced_context.items[0].key == "rules"


def test_token_budget_is_primary_when_present() -> None:
    optimizer = ContextOptimizer(token_counter=WordTokenCounter())
    result = optimizer.optimize(
        ContextRequest(
            context=(
                ContextItem("required", "one two", required=True),
                ContextItem("high", "three four five", priority=90),
                ContextItem("low", "six seven eight nine ten", priority=10),
            ),
            context_budget_chars=100000,
            context_budget_tokens=5,
        )
    )

    assert tuple(item.key for item in result.reduced_context.items) == (
        "required",
        "high",
    )
    assert result.reduced_context.omitted_keys == ("low",)
    assert result.original_context_tokens == 10
    assert result.reduced_context_tokens == 5
    assert result.saved_context_tokens == 5
    assert result.token_reduction_ratio == 0.5


def test_custom_token_counter_can_be_injected() -> None:
    optimizer = ContextOptimizer(token_counter=WordTokenCounter())
    result = optimizer.optimize(
        ContextRequest(
            context=(ContextItem("a", "one two three"),),
            context_budget_tokens=10,
        )
    )

    assert result.original_context_tokens == 3
    assert result.reduced_context_tokens == 3


def test_default_counter_is_conservative_for_non_ascii() -> None:
    counter = ApproxTokenCounter()
    assert counter.count("가나다") == 3
    assert counter.count("abcdefgh") == 2


def test_cli_payload_contains_only_context_data() -> None:
    request = _parse_request(
        {
            "context_budget_tokens": 10,
            "context": [
                {"key": "rules", "content": "Rule A", "required": True},
                {"key": "noise", "content": "N" * 3000, "priority": 1},
            ],
        }
    )
    assert request.context_budget_tokens == 10

    payload = _serialize(ContextOptimizer().optimize(request))
    assert set(payload) == {
        "context",
        "omitted_keys",
        "duplicate_keys",
        "unchanged_keys",
        "superseded_keys",
        "compacted_keys",
        "chunked_keys",
        "context_fingerprints",
        "original_context_chars",
        "reduced_context_chars",
        "reduction_ratio",
        "original_context_tokens",
        "reduced_context_tokens",
        "saved_context_tokens",
        "token_reduction_ratio",
        "stage_token_savings",
        "notes",
    }
    assert all("prompt" not in key for key in payload)



def test_explicit_state_supersession_keeps_latest_version() -> None:
    result = ContextOptimizer().optimize(
        ContextRequest(
            context=(
                ContextItem(
                    "state-v1",
                    "old state",
                    state_slot="worker-state",
                    state_version=1,
                    priority=100,
                ),
                ContextItem(
                    "state-v2",
                    "current state",
                    state_slot="worker-state",
                    state_version=2,
                    priority=100,
                ),
            ),
            context_budget_tokens=100,
        )
    )

    assert tuple(item.key for item in result.reduced_context.items) == ("state-v2",)
    assert result.reduced_context.superseded_keys == ("state-v1",)


def test_density_selection_prefers_more_value_per_token() -> None:
    optimizer = ContextOptimizer(token_counter=WordTokenCounter())
    result = optimizer.optimize(
        ContextRequest(
            context=(
                ContextItem(
                    "large",
                    " ".join(["large"] * 9),
                    priority=90,
                ),
                ContextItem(
                    "small-a",
                    "small useful",
                    priority=70,
                ),
                ContextItem(
                    "small-b",
                    "tiny useful",
                    priority=65,
                ),
            ),
            context_budget_tokens=5,
            selection_strategy="density",
        )
    )

    assert tuple(item.key for item in result.reduced_context.items) == (
        "small-a",
        "small-b",
    )
    assert "large" in result.reduced_context.omitted_keys


def test_opt_in_block_dedup_compacts_exact_repeated_blocks() -> None:
    optimizer = ContextOptimizer(token_counter=WordTokenCounter())
    result = optimizer.optimize(
        ContextRequest(
            context=(
                ContextItem(
                    "log",
                    "same line\nsame line\nunique line",
                    required=True,
                    dedup_blocks=True,
                ),
            ),
            context_budget_tokens=100,
        )
    )

    item = result.reduced_context.items[0]
    assert item.content == "same line\nunique line"
    assert result.reduced_context.compacted_keys == ("log",)
    assert result.original_context_tokens == 6
    assert result.reduced_context_tokens == 4



def test_stage_savings_add_up_to_total_estimated_savings() -> None:
    optimizer = ContextOptimizer(token_counter=WordTokenCounter())
    result = optimizer.optimize(
        ContextRequest(
            context=(
                ContextItem(
                    "log",
                    "repeat line\nrepeat line\nunique",
                    dedup_blocks=True,
                    priority=20,
                ),
                ContextItem(
                    "copy",
                    "unique",
                    priority=1,
                ),
                ContextItem(
                    "state-v1",
                    "old state",
                    state_slot="state",
                    state_version=1,
                    priority=10,
                ),
                ContextItem(
                    "state-v2",
                    "new state",
                    state_slot="state",
                    state_version=2,
                    required=True,
                    priority=100,
                ),
            ),
            context_budget_tokens=10,
        )
    )

    assert sum(result.stage_token_savings.values()) == result.saved_context_tokens
    assert result.stage_token_savings["block_dedup"] > 0
    assert result.stage_token_savings["supersession"] > 0



def test_whitespace_sensitive_content_is_not_deduplicated() -> None:
    result = ContextOptimizer().optimize(
        ContextRequest(
            context=(
                ContextItem("code-a", "if ready:\n    run()"),
                ContextItem("code-b", "if ready:\n        run()"),
            ),
            context_budget_tokens=100,
        )
    )

    assert result.reduced_context.duplicate_keys == ()
    assert tuple(item.key for item in result.reduced_context.items) == (
        "code-a",
        "code-b",
    )


def test_duplicate_keys_are_rejected_before_fingerprint_overwrite() -> None:
    import pytest

    with pytest.raises(ValueError, match="duplicate ContextItem key"):
        ContextOptimizer().optimize(
            ContextRequest(
                context=(
                    ContextItem("same-key", "A"),
                    ContextItem("same-key", "B"),
                )
            )
        )


def test_balanced_selection_preserves_priority_band_then_density() -> None:
    optimizer = ContextOptimizer(token_counter=WordTokenCounter())
    result = optimizer.optimize(
        ContextRequest(
            context=(
                ContextItem(
                    "large-high",
                    "one two three four five",
                    priority=91,
                ),
                ContextItem(
                    "small-high",
                    "six seven",
                    priority=90,
                ),
                ContextItem(
                    "medium",
                    "eight nine ten eleven",
                    priority=89,
                ),
            ),
            context_budget_tokens=6,
            selection_strategy="balanced",
        )
    )

    assert tuple(item.key for item in result.reduced_context.items) == (
        "small-high",
        "medium",
    )
    assert result.reduced_context.omitted_keys == ("large-high",)


def test_token_aware_chunking_splits_only_opt_in_optional_items() -> None:
    optimizer = ContextOptimizer(token_counter=WordTokenCounter())
    result = optimizer.optimize(
        ContextRequest(
            context=(
                ContextItem(
                    "history",
                    "alpha one\n\nbeta two\n\ngamma three",
                    priority=20,
                    chunk_mode="paragraph",
                    chunk_target_tokens=3,
                ),
            ),
            context_budget_tokens=4,
        )
    )

    assert result.reduced_context.chunked_keys == ("history",)
    assert tuple(item.key for item in result.reduced_context.items) == (
        "history::chunk:001",
        "history::chunk:002",
    )
    assert result.reduced_context.omitted_keys == (
        "history::chunk:003",
    )
    assert all(
        item.chunk_mode is None
        for item in result.reduced_context.items
    )


def test_required_item_is_never_partially_chunked() -> None:
    optimizer = ContextOptimizer(token_counter=WordTokenCounter())
    result = optimizer.optimize(
        ContextRequest(
            context=(
                ContextItem(
                    "required-history",
                    "alpha one\n\nbeta two\n\ngamma three",
                    required=True,
                    chunk_mode="paragraph",
                    chunk_target_tokens=2,
                ),
            ),
            context_budget_tokens=2,
        )
    )

    assert result.reduced_context.chunked_keys == ()
    assert result.reduced_context.items[0].key == "required-history"
    assert result.reduced_context.items[0].content.count("\n\n") == 2
