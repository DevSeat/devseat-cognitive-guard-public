from __future__ import annotations

from pathlib import Path

from devseat_cognitive_guard import (
    ContextItem,
    OptimizationProfile,
)
from devseat_cognitive_guard.policy_store import (
    FilePolicyStore,
    PolicyNamespace,
)
from devseat_cognitive_guard.runtime import ProductContextRuntime
from devseat_cognitive_guard.token_adapters import CallableTokenCounter


def _counter():
    return CallableTokenCounter(lambda text: len(text.split()))


def test_policy_store_isolates_products_and_profiles(tmp_path: Path) -> None:
    store = FilePolicyStore(tmp_path)
    bio = PolicyNamespace("example-bio", "default")
    agent = PolicyNamespace("example-agent", "default")

    bio_policy = store.load(bio)
    item = ContextItem(
        "memory-001",
        "optional",
        source="memory",
        priority=10,
    )
    bio_policy.observe(
        items=(item,),
        omitted_keys=(item.key,),
        raw_quality_pass=True,
        optimized_quality_pass=False,
    )
    store.save(bio, bio_policy)

    loaded_bio = store.load(bio)
    loaded_agent = store.load(agent)

    assert loaded_bio.priority_bias(item) > 0
    assert loaded_agent.priority_bias(item) == 0
    assert store.path_for(bio) != store.path_for(agent)
    assert "example" not in str(store.path_for(bio).relative_to(tmp_path))


def test_product_runtime_persists_learning_across_restart(tmp_path: Path) -> None:
    store = FilePolicyStore(tmp_path)
    items = (
        ContextItem(
            "critical-001",
            "critical fact",
            priority=1,
            source="critical",
        ),
        ContextItem(
            "noise-001",
            "noise fact",
            priority=10,
            source="noise",
        ),
    )
    profile = OptimizationProfile(
        token_budget=2,
        selection_strategy="balanced",
    )
    runtime = ProductContextRuntime(
        product="BIO",
        policy_store=store,
        token_counter=_counter(),
        default_profile=profile,
    )
    first = runtime.optimize(items)
    assert "critical-001" in first.omitted_keys

    runtime.observe(
        items=items,
        result=first,
        raw_quality_pass=True,
        optimized_quality_pass=False,
    )

    restarted = ProductContextRuntime(
        product="BIO",
        policy_store=store,
        token_counter=_counter(),
        default_profile=profile,
    )
    assert restarted.policy.priority_bias(items[0]) > 0


def test_product_runtime_reset_only_resets_its_namespace(tmp_path: Path) -> None:
    store = FilePolicyStore(tmp_path)
    one = ProductContextRuntime(
        product="one",
        policy_store=store,
        token_counter=_counter(),
    )
    two = ProductContextRuntime(
        product="two",
        policy_store=store,
        token_counter=_counter(),
    )
    item = ContextItem("x-001", "x", source="shared")

    for runtime in (one, two):
        runtime.policy.observe(
            items=(item,),
            omitted_keys=(item.key,),
            raw_quality_pass=True,
            optimized_quality_pass=False,
        )
        runtime.save()

    one.reset_policy()

    assert one.policy.priority_bias(item) == 0
    assert FilePolicyStore(tmp_path).load(
        PolicyNamespace("two")
    ).priority_bias(item) > 0
