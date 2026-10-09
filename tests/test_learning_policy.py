from pathlib import Path

from devseat_cognitive_guard import (
    AdaptiveContextPolicy,
    ContextItem,
    ContextOptimizer,
    ContextRequest,
)
from devseat_cognitive_guard.learning_workload import (
    build_generalization_challenge_workload,
)


class WordTokenCounter:
    def count(self, text: str) -> int:
        return len(text.split())


def test_regression_increases_keep_bias() -> None:
    policy = AdaptiveContextPolicy()
    item = ContextItem(
        "old-log-01",
        "optional evidence",
        source="logs",
        priority=10,
    )

    touched = policy.observe(
        items=(item,),
        omitted_keys=("old-log-01",),
        raw_quality_pass=True,
        optimized_quality_pass=False,
    )

    assert touched >= 1
    assert policy.priority_bias(item) > 0


def test_safe_omissions_slowly_reduce_keep_bias() -> None:
    policy = AdaptiveContextPolicy()
    item = ContextItem(
        "old-log-01",
        "optional evidence",
        source="logs",
        priority=10,
    )

    for _ in range(3):
        policy.observe(
            items=(item,),
            omitted_keys=("old-log-01",),
            raw_quality_pass=True,
            optimized_quality_pass=True,
        )

    assert policy.priority_bias(item) < 0


def test_raw_failure_does_not_train_policy() -> None:
    policy = AdaptiveContextPolicy()
    item = ContextItem("optional", "evidence", source="logs")

    touched = policy.observe(
        items=(item,),
        omitted_keys=("optional",),
        raw_quality_pass=False,
        optimized_quality_pass=False,
    )

    assert touched == 0
    assert policy.priority_bias(item) == 0


def test_required_item_is_never_reprioritized() -> None:
    policy = AdaptiveContextPolicy()
    item = ContextItem(
        "required",
        "must keep",
        source="logs",
        priority=1,
        required=True,
    )
    policy.observe(
        items=(ContextItem("other", "x", source="logs"),),
        omitted_keys=("other",),
        raw_quality_pass=True,
        optimized_quality_pass=False,
    )

    adjusted = policy.apply((item,))
    assert adjusted[0].priority == 1
    assert adjusted[0].required is True


def test_policy_changes_future_optional_selection() -> None:
    policy = AdaptiveContextPolicy()
    risky = ContextItem(
        "risky-01",
        "risky evidence",
        source="risky-source",
        priority=40,
    )
    ordinary = ContextItem(
        "ordinary-01",
        "ordinary evidence",
        source="ordinary-source",
        priority=45,
    )

    # Teach the policy that omitting risky-source caused regressions.
    for _ in range(2):
        policy.observe(
            items=(risky,),
            omitted_keys=(risky.key,),
            raw_quality_pass=True,
            optimized_quality_pass=False,
        )

    optimizer = ContextOptimizer(
        token_counter=WordTokenCounter(),
        learning_policy=policy,
    )
    result = optimizer.optimize(
        ContextRequest(
            context=(risky, ordinary),
            context_budget_tokens=2,
        )
    )

    assert tuple(item.key for item in result.reduced_context.items) == (
        "risky-01",
    )


def test_policy_round_trip_does_not_store_context_text(tmp_path: Path) -> None:
    policy = AdaptiveContextPolicy()
    item = ContextItem(
        "log-123",
        "SECRET CONTEXT TEXT",
        source="runtime-log",
        state_slot="worker-state",
    )
    policy.observe(
        items=(item,),
        omitted_keys=(item.key,),
        raw_quality_pass=True,
        optimized_quality_pass=False,
    )

    path = tmp_path / "policy.json"
    policy.save(path)
    raw = path.read_text(encoding="utf-8")
    loaded = AdaptiveContextPolicy.load(path)

    assert "SECRET CONTEXT TEXT" not in raw
    assert loaded.priority_bias(item) == policy.priority_bias(item)



def test_generalization_workload_switches_source_but_keeps_key_family() -> None:
    cases = build_generalization_challenge_workload(
        turns=8,
        source_switch_turn=5,
        decoys_per_turn=3,
    )

    train_item = cases[0]["context"][-1]
    eval_item = cases[-1]["context"][-1]

    assert train_item["source"] == "evidence-train"
    assert eval_item["source"] == "evidence-unseen"
    assert train_item["key"].startswith("critical-evidence-")
    assert eval_item["key"].startswith("critical-evidence-")



def test_targeted_attribution_only_updates_causal_feature() -> None:
    policy = AdaptiveContextPolicy()
    critical = ContextItem(
        "critical-001",
        "critical",
        source="critical-source",
        priority=1,
    )
    noise = ContextItem(
        "noise-001",
        "noise",
        source="noise-source",
        priority=1,
    )

    policy.observe(
        items=(critical, noise),
        omitted_keys=(critical.key, noise.key),
        raw_quality_pass=True,
        optimized_quality_pass=False,
        attributed_keys=(critical.key,),
    )

    assert policy.priority_bias(critical) > 0
    assert policy.priority_bias(noise) == 0
