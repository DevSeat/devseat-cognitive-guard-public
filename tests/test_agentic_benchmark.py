from devseat_cognitive_guard.agentic_benchmark import (
    REQUIRED_EVIDENCE,
    AgenticToolEnvironment,
)
from devseat_cognitive_guard.agentic_workload import iter_agentic_hell_workload


def _case(turn: int):
    return list(
        iter_agentic_hell_workload(
            turns=turn,
            noise_blocks_per_turn=1,
            noise_chars=128,
        )
    )[-1]


def _gather_required_evidence(env: AgenticToolEnvironment, case) -> None:
    for group in ("code", "deployment", "dependency"):
        result = env.execute(
            "read_current_state",
            {"group": group, "consistency": "linearizable"},
        )
        assert result["status"] == "ok"
        assert result["authoritative"] is True

    for name in ("rollout", "approval", "release"):
        result = env.execute(
            "lookup_policy",
            {"name": name, "version": "active"},
        )
        assert result["status"] == "ok"
        assert result["version"] == "active"

    dependency = env.execute(
        "inspect_dependency",
        {"service": case["state"]["upstream"]},
    )
    if dependency["status"] == "transient_error":
        dependency = env.execute(
            "inspect_dependency",
            {"service": case["state"]["upstream"]},
        )
    assert dependency["status"] == "ok"

    artifact = env.execute(
        "read_artifact",
        {
            "path": case["state"]["artifact_path"],
            "freshness": "fresh",
        },
    )
    assert artifact["status"] == "ok"
    assert artifact["authoritative"] is True

    calculation = env.execute(
        "calculator",
        {
            "operation": "majority_quorum",
            "value": case["artifact"]["shard_count"],
        },
    )
    assert calculation["status"] == "ok"


def test_agentic_workload_is_cumulative_and_keeps_current_truth_out_of_history() -> None:
    cases = list(
        iter_agentic_hell_workload(
            turns=3,
            noise_blocks_per_turn=2,
            noise_chars=128,
        )
    )

    assert len(cases) == 3
    assert len(cases[0]["context"]) < len(cases[1]["context"]) < len(cases[2]["context"])
    assert cases[2]["state"]["checkpoint"] == "decision-0003"
    assert cases[2]["expected"]["telemetry_port"] == 4317
    assert cases[2]["expected"]["timeout"] == "12s"
    assert cases[2]["expected"]["quorum"] == (
        cases[2]["artifact"]["shard_count"] // 2 + 1
    )


def test_linearizable_state_is_authoritative_but_eventual_is_trap() -> None:
    case = _case(2)
    env = AgenticToolEnvironment(case)

    eventual = env.execute(
        "read_current_state",
        {"group": "deployment", "consistency": "eventual"},
    )
    assert eventual["authoritative"] is False
    assert eventual["checkpoint"] == "decision-0001"
    assert env.trap_calls == 1
    assert "state:deployment" not in env.evidence

    current = env.execute(
        "read_current_state",
        {"group": "deployment", "consistency": "linearizable"},
    )
    assert current["authoritative"] is True
    assert current["checkpoint"] == "decision-0002"
    assert "state:deployment" in env.evidence


def test_turn_seven_dependency_requires_recovery_retry() -> None:
    case = _case(7)
    env = AgenticToolEnvironment(case)

    first = env.execute(
        "inspect_dependency",
        {"service": case["state"]["upstream"]},
    )
    assert first["status"] == "transient_error"
    assert first["retryable"] is True
    assert "dependency" not in env.evidence

    second = env.execute(
        "inspect_dependency",
        {"service": case["state"]["upstream"]},
    )
    assert second["status"] == "ok"
    assert "dependency" in env.evidence


def test_untrusted_tool_text_is_present_on_adversarial_turns() -> None:
    case11 = _case(11)
    env11 = AgenticToolEnvironment(case11)
    dependency = env11.execute(
        "inspect_dependency",
        {"service": case11["state"]["upstream"]},
    )
    assert "UNTRUSTED SERVICE BANNER" in dependency["banner"]

    case13 = _case(13)
    env13 = AgenticToolEnvironment(case13)
    artifact = env13.execute(
        "read_artifact",
        {
            "path": case13["state"]["artifact_path"],
            "freshness": "fresh",
        },
    )
    assert "UNTRUSTED ARTIFACT NOTE" in artifact["note"]


def test_submit_rejects_correct_answer_without_required_tool_evidence() -> None:
    case = _case(3)
    env = AgenticToolEnvironment(case)

    result = env.execute(
        "submit_release_decision",
        case["expected"],
    )

    assert result["status"] == "rejected"
    assert result["reason"] == "missing_authoritative_evidence"
    assert set(result["missing"]) == REQUIRED_EVIDENCE


def test_submit_accepts_only_after_full_authoritative_evidence() -> None:
    case = _case(4)
    env = AgenticToolEnvironment(case)
    _gather_required_evidence(env, case)

    assert env.evidence_coverage == 1.0
    result = env.execute(
        "submit_release_decision",
        case["expected"],
    )

    assert result["status"] == "accepted"
    assert env.first_submit_accepted is True
    assert env.final_submission == case["expected"]


def test_legacy_cache_and_history_are_counted_as_traps() -> None:
    case = _case(2)
    env = AgenticToolEnvironment(case)

    env.execute("read_cached_state", {"group": "code"})
    env.execute(
        "lookup_policy",
        {"name": "approval", "version": "legacy"},
    )
    history = env.execute("search_history", {"query": "checkpoint"})

    assert env.trap_calls == 3
    assert history["authoritative"] is False
    assert "UNTRUSTED ARCHIVE TEXT" in history["archived_note"]
