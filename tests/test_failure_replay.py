from devseat_cognitive_guard import ContextItem
from devseat_cognitive_guard.failure_replay import (
    attribute_omission_failure,
    make_replay_record,
)


def test_failure_replay_finds_single_recovering_item() -> None:
    def probe(restored: tuple[str, ...]) -> bool:
        return "critical" in restored

    result = attribute_omission_failure(
        ("noise-a", "critical", "noise-b", "noise-c"),
        probe=probe,
        max_probes=8,
    )

    assert result.recovered is True
    assert result.attributed_keys == ("critical",)
    assert result.probes <= 8


def test_failure_replay_reports_unresolved_when_full_restore_fails() -> None:
    result = attribute_omission_failure(
        ("a", "b"),
        probe=lambda restored: False,
        max_probes=4,
    )

    assert result.recovered is False
    assert result.attributed_keys == ()
    assert result.probes == 1


def test_failure_replay_keeps_interaction_group() -> None:
    def probe(restored: tuple[str, ...]) -> bool:
        return "a" in restored and "b" in restored

    result = attribute_omission_failure(
        ("a", "b"),
        probe=probe,
        max_probes=8,
    )

    assert result.recovered is True
    assert set(result.attributed_keys) == {"a", "b"}
    assert result.interaction_group is True


def test_replay_record_contains_metadata_features_not_content() -> None:
    items = (
        ContextItem(
            "critical-001",
            "private content",
            source="runtime",
            state_slot="worker",
        ),
        ContextItem("noise-001", "other content", source="noise"),
    )
    attribution = attribute_omission_failure(
        ("critical-001", "noise-001"),
        probe=lambda restored: "critical-001" in restored,
        max_probes=8,
    )
    record = make_replay_record(
        case_id="case-1",
        token_budget=128,
        selection_strategy="priority",
        attribution=attribution,
        items=items,
        omitted_keys=("critical-001", "noise-001"),
    )

    rendered = str(record)
    assert "source:runtime" in rendered
    assert "slot:worker" in rendered
    assert "private content" not in rendered
