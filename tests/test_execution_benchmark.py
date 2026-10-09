from devseat_cognitive_guard.agentic_workload import iter_agentic_hell_workload
from devseat_cognitive_guard.execution_benchmark import (
    TracedEnvironment,
    _authoritative_state,
    _compress_evidence,
    run_execution_case,
)


class FakeExecutionClient:
    def __init__(self, case) -> None:
        self.case = case
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
        if self.calls == 1:
            tool_calls = [
                _tool_call(
                    "state-code",
                    "read_current_state",
                    {"group": "code", "consistency": "linearizable"},
                ),
                _tool_call(
                    "state-deployment",
                    "read_current_state",
                    {"group": "deployment", "consistency": "linearizable"},
                ),
                _tool_call(
                    "state-dependency",
                    "read_current_state",
                    {"group": "dependency", "consistency": "linearizable"},
                ),
                _tool_call(
                    "policy-rollout",
                    "lookup_policy",
                    {"name": "rollout", "version": "active"},
                ),
                _tool_call(
                    "policy-approval",
                    "lookup_policy",
                    {"name": "approval", "version": "active"},
                ),
                _tool_call(
                    "policy-release",
                    "lookup_policy",
                    {"name": "release", "version": "active"},
                ),
            ]
        else:
            tool_calls = [
                _tool_call(
                    "submit",
                    "submit_release_decision",
                    self.case["expected"],
                )
            ]

        return (
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": tool_calls,
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                    "total_tokens": 1100,
                },
            },
            {},
            1000.0,
        )


def _tool_call(call_id, name, arguments):
    import json

    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def _case(turn=1):
    return list(
        iter_agentic_hell_workload(
            turns=turn,
            noise_blocks_per_turn=1,
            noise_chars=128,
        )
    )[-1]


def test_authoritative_state_is_built_from_tool_outputs() -> None:
    case = _case(2)
    env = TracedEnvironment(case)

    for group in ("code", "deployment", "dependency"):
        env.execute_traced(
            "read_current_state",
            {"group": group, "consistency": "linearizable"},
            source="test",
        )

    state = _authoritative_state(env)

    assert state["module"] == case["state"]["module"]
    assert state["checkpoint"] == case["state"]["checkpoint"]
    assert state["upstream"] == case["state"]["upstream"]
    assert state["artifact_path"] == case["state"]["artifact_path"]


def test_compressed_evidence_uses_real_tool_outputs() -> None:
    case = _case(3)
    env = TracedEnvironment(case)

    for group in ("code", "deployment", "dependency"):
        env.execute_traced(
            "read_current_state",
            {"group": group, "consistency": "linearizable"},
            source="test",
        )
    for name in ("rollout", "approval", "release"):
        env.execute_traced(
            "lookup_policy",
            {"name": name, "version": "active"},
            source="test",
        )

    env.execute_traced(
        "inspect_dependency",
        {"service": case["state"]["upstream"]},
        source="test",
    )
    artifact = env.execute_traced(
        "read_artifact",
        {
            "path": case["state"]["artifact_path"],
            "freshness": "fresh",
        },
        source="test",
    )
    env.execute_traced(
        "calculator",
        {
            "operation": "majority_quorum",
            "value": artifact["shard_count"],
        },
        source="test",
    )

    evidence = _compress_evidence(env)

    assert evidence["state"]["checkpoint"] == case["state"]["checkpoint"]
    assert evidence["dependency"]["health"] == case["dependency"]["status"]
    assert evidence["artifact"]["build_id"] == case["artifact"]["build_id"]
    assert evidence["calculation"]["result"] == case["expected"]["quorum"]
    assert evidence["evidence_coverage"] == 1.0


def test_two_call_execution_path_accepts_cleanly() -> None:
    case = _case(4)
    client = FakeExecutionClient(case)

    result = run_execution_case(
        client,
        case,
        model="gpt-6-luna",
        token_budget=500,
        selection_strategy="balanced",
        max_output_tokens=4096,
        reasoning_effort="xhigh",
        reasoning_mode="pro",
        service_tier=None,
    )

    assert result.accepted is True
    assert result.clean_pass is True
    assert result.first_submit_accepted is True
    assert result.model_calls == 2
    assert result.submit_attempts == 1
    assert result.trap_calls == 0
    assert result.protocol_violations == 0
    assert result.evidence_coverage == 1.0
    assert result.total_tokens == 2200
    assert result.compressed_evidence_tokens > 0


def test_transient_dependency_failure_is_retried_without_model_roundtrip() -> None:
    case = _case(7)
    client = FakeExecutionClient(case)

    result = run_execution_case(
        client,
        case,
        model="gpt-6-luna",
        token_budget=500,
        selection_strategy="balanced",
        max_output_tokens=4096,
        reasoning_effort="xhigh",
        reasoning_mode="pro",
        service_tier=None,
    )

    assert result.accepted is True
    assert result.model_calls == 2
    assert result.tool_errors == 1
    classifications = [item.classification for item in result.tool_trace]
    sources = [item.source for item in result.tool_trace]
    assert "retryable" in classifications
    assert "optimizer-retry" in sources
