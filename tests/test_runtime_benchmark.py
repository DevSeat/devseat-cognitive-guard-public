import json

from devseat_cognitive_guard.agentic_workload import iter_agentic_hell_workload
from devseat_cognitive_guard.runtime_benchmark import (
    _project_evidence,
    _stable_prefix_fingerprint,
    run_runtime_case,
)


class FakeRuntimeClient:
    def __init__(self, case, *, reject_first=False) -> None:
        self.case = case
        self.reject_first = reject_first
        self.calls = 0

    def chat_completions_create(
        self,
        *,
        model,
        messages,
        tools,
        max_output_tokens,
        reasoning_effort,
        reasoning_mode,
        service_tier,
    ):
        self.calls += 1
        submission = dict(self.case["expected"])
        if self.reject_first and self.calls == 1:
            submission["release_action"] = "wrong-action"

        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": f"submit-{self.calls}",
                                "type": "function",
                                "function": {
                                    "name": "submit_release_decision",
                                    "arguments": json.dumps(submission),
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 800,
                "completion_tokens": 80,
                "total_tokens": 880,
            },
        }
        return payload, {}, 900.0


def _case(turn=1):
    return list(
        iter_agentic_hell_workload(
            turns=turn,
            noise_blocks_per_turn=1,
            noise_chars=128,
        )
    )[-1]


def _run(case, client):
    return run_runtime_case(
        client,
        case,
        model="gpt-6-luna",
        token_budget=500,
        selection_strategy="balanced",
        max_output_tokens=2048,
        base_reasoning_effort="medium",
        retry_reasoning_effort="high",
        repair_reasoning_effort="xhigh",
        reasoning_mode="pro",
        service_tier=None,
    )


def test_known_toolgraph_uses_one_model_call_on_clean_path() -> None:
    case = _case(4)
    result = _run(case, FakeRuntimeClient(case))

    assert result.accepted is True
    assert result.clean_pass is True
    assert result.first_submit_accepted is True
    assert result.model_calls == 1
    assert result.submit_attempts == 1
    assert result.evidence_coverage == 1.0
    assert result.selected_reasoning_effort == "medium"
    assert result.repair_used is False
    assert result.total_tokens == 880


def test_retryable_tool_failure_escalates_reasoning_without_planner_call() -> None:
    case = _case(7)
    result = _run(case, FakeRuntimeClient(case))

    assert result.accepted is True
    assert result.clean_pass is True
    assert result.model_calls == 1
    assert result.selected_reasoning_effort == "high"
    assert result.tool_errors == 1
    assert any(
        item.classification == "retryable"
        for item in result.tool_trace
    )
    assert any(
        item.source == "runtime-local-retry"
        for item in result.tool_trace
    )


def test_rejected_first_submit_uses_xhigh_repair_call() -> None:
    case = _case(5)
    result = _run(
        case,
        FakeRuntimeClient(case, reject_first=True),
    )

    assert result.accepted is True
    assert result.clean_pass is False
    assert result.first_submit_accepted is False
    assert result.model_calls == 2
    assert result.submit_attempts == 2
    assert result.repair_used is True
    assert result.repair_reasoning_effort == "xhigh"
    assert result.total_tokens == 1760


def test_projected_evidence_keeps_only_decision_relevant_policy_values() -> None:
    evidence = {
        "state": {
            "module": "m",
            "checkpoint": "c",
            "branch": "b",
            "schema": "s",
            "ring": "canary",
            "risk": "high",
            "region": "kr-central",
            "upstream": "svc",
        },
        "policies": {
            "rollout": {
                "canary": {
                    "low": "lane-c1",
                    "medium": "lane-c1",
                    "high": "lane-c2",
                    "critical": "lane-c2",
                },
                "stable": {
                    "low": "lane-s1",
                    "medium": "lane-s1",
                    "high": "lane-s2",
                    "critical": "lane-s2",
                },
            },
            "approval": {
                "low": "auto-approve",
                "medium": "peer-review",
                "high": "dual-review",
                "critical": "security-board",
            },
            "release": [
                "if dependency_status=degraded OR risk=critical -> hold",
                "else if dependency_status=warning OR risk=high -> review",
                "else -> proceed",
            ],
        },
        "dependency": {
            "service": "svc",
            "health": "healthy",
            "latency_ms": 10,
        },
        "artifact": {
            "build_id": "build-1",
            "shard_count": 5,
            "content_digest": "unused",
        },
        "calculation": {
            "operation": "majority_quorum",
            "input": 5,
            "result": 3,
        },
        "evidence_coverage": 1.0,
    }

    projected = _project_evidence(evidence)

    assert projected["policy_projection"]["rollout_lane"] == "lane-c2"
    assert projected["policy_projection"]["approval_gate"] == "dual-review"
    assert projected["dependency_status"] == "healthy"
    assert projected["artifact"] == {
        "build_id": "build-1",
        "shard_count": 5,
    }
    assert projected["calculated_quorum"] == 3
    assert "content_digest" not in json.dumps(projected)
    assert "stable" not in json.dumps(projected)


def test_stable_prefix_fingerprint_is_deterministic() -> None:
    assert _stable_prefix_fingerprint() == _stable_prefix_fingerprint()
    assert len(_stable_prefix_fingerprint()) == 16
