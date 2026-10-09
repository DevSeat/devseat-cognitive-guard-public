from pathlib import Path

from devseat_cognitive_guard.pleum_benchmark import (
    PleumRouterClient,
    _append_checkpoint,
    _build_input,
    _extract_output_text,
    _load_checkpoint,
    _parse_budgets,
    _quality_pass,
    _slice_cases,
    build_markdown_report,
    long_task_progression,
    run_case,
    summarize,
    summarize_by_budget,
)
from devseat_cognitive_guard.long_workload import (
    build_long_workload,
    iter_long_workload,
)


class FakePleumClient:
    def __init__(self) -> None:
        self.calls = []

    def responses_create(
        self,
        *,
        model,
        task,
        context_text,
        max_output_tokens,
    ):
        self.calls.append(
            {
                "model": model,
                "task": task,
                "context_text": context_text,
                "max_output_tokens": max_output_tokens,
            }
        )
        raw = len(self.calls) == 1
        input_tokens = 100 if raw else 60
        return (
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Do not change public APIs.",
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 8,
                    "total_tokens": input_tokens + 8,
                },
            },
            {
                "X-Cost-Krw": "10" if raw else "6",
                "X-Cost-Usd": "0.01" if raw else "0.006",
            },
            123.0 if raw else 111.0,
        )


def test_benchmark_compares_actual_router_usage() -> None:
    client = FakePleumClient()
    result = run_case(
        client,
        {
            "id": "duplicate",
            "task": "What must not be changed?",
            "expected_contains": ["public APIs"],
            "context": [
                {
                    "key": "a",
                    "content": "Do not change public APIs.",
                    "required": True,
                    "priority": 100,
                },
                {
                    "key": "b",
                    "content": "Do not change public APIs.",
                    "priority": 1,
                },
            ],
        },
        model="gpt-6-luna",
        token_budget=None,
        max_output_tokens=128,
    )

    assert result.raw.input_tokens == 100
    assert result.optimized.input_tokens == 60
    assert result.actual_saved_input_tokens == 40
    assert result.actual_reduction_ratio == 0.4
    assert result.raw.quality_pass is True
    assert result.optimized.quality_pass is True
    assert result.duplicate_keys == ("b",)
    assert len(client.calls) == 2


def test_summary_uses_actual_tokens_and_costs() -> None:
    client = FakePleumClient()
    result = run_case(
        client,
        {
            "id": "one",
            "task": "What must not be changed?",
            "expected_contains": ["public APIs"],
            "context": [
                {
                    "key": "a",
                    "content": "Do not change public APIs.",
                    "required": True,
                }
            ],
        },
        model="gpt-6-luna",
        token_budget=None,
        max_output_tokens=128,
    )

    summary = summarize([result])
    assert summary["raw_input_tokens"] == 100
    assert summary["optimized_input_tokens"] == 60
    assert summary["saved_input_tokens"] == 40
    assert summary["saved_cost_krw"] == 4.0


def test_quality_validator_is_case_insensitive() -> None:
    assert _quality_pass("PUBLIC apis must remain.", ("public APIs",))
    assert not _quality_pass("No answer.", ("public APIs",))
    assert not _quality_pass(
        "Current checkpoint-02, old checkpoint-01.",
        ("checkpoint-02",),
        forbidden=("checkpoint-0001",),
    )


def test_quality_validator_normalizes_zero_padded_identifier_suffixes() -> None:
    assert _quality_pass(
        "Current checkpoint-1.",
        ("checkpoint-0001",),
    )
    assert not _quality_pass(
        "Current checkpoint-2, old checkpoint-01.",
        ("checkpoint-0002",),
        forbidden=("checkpoint-0001",),
    )


def test_quality_validator_does_not_match_identifier_prefixes() -> None:
    assert _quality_pass(
        "Current worker.py checkpoint-10 telemetry port 4317.",
        ("worker.py", "checkpoint-0010", "4317"),
        forbidden=("checkpoint-0001",),
    )
    assert _quality_pass(
        "Current checkpoint-100.",
        ("checkpoint-0100",),
        forbidden=("checkpoint-0010", "checkpoint-0001"),
    )


def test_quality_validator_does_not_rewrite_standalone_numbers() -> None:
    assert not _quality_pass(
        "Value is 1.",
        ("001",),
    )


def test_output_text_extraction() -> None:
    assert (
        _extract_output_text(
            {
                "output": [
                    {
                        "content": [
                            {"type": "output_text", "text": "hello"},
                            {"type": "output_text", "text": " world"},
                        ]
                    }
                ]
            }
        )
        == "hello world"
    )


def test_raw_and_optimized_prompt_wrapper_is_identical() -> None:
    first = _build_input(task="Same task.", context_text="AAA_CONTEXT")
    second = _build_input(task="Same task.", context_text="BBB_CONTEXT")
    assert first.replace("AAA_CONTEXT", "CTX") == second.replace(
        "BBB_CONTEXT",
        "CTX",
    )



def test_long_workload_accumulates_context() -> None:
    cases = build_long_workload(
        turns=4,
        noise_blocks_per_turn=2,
        noise_chars=80,
    )

    assert len(cases) == 4
    assert cases[0]["turn"] == 1
    assert cases[-1]["turn"] == 4
    assert len(cases[-1]["context"]) > len(cases[0]["context"])
    assert cases[-1]["group"] == "long-task"
    assert "checkpoint-0004" in cases[-1]["expected_contains"]
    assert "checkpoint-0001" in cases[-1]["expected_not_contains"]
    assert "checkpoint-0003" in cases[-1]["expected_not_contains"]
    state_items = [
        item
        for item in cases[-1]["context"]
        if item.get("state_slot") == "engineering-state"
    ]
    assert len(state_items) == 4
    assert state_items[-1]["state_version"] == 4


def test_budget_sweep_parser() -> None:
    assert _parse_budgets(None, 24) == [24]
    assert _parse_budgets("16,24,32", None) == [16, 24, 32]


def test_summary_by_budget_separates_sweep_results() -> None:
    client_a = FakePleumClient()
    result_16 = run_case(
        client_a,
        {
            "id": "one",
            "task": "What must not be changed?",
            "expected_contains": ["public APIs"],
            "context": [
                {
                    "key": "a",
                    "content": "Do not change public APIs.",
                    "required": True,
                }
            ],
        },
        model="gpt-6-luna",
        token_budget=16,
        max_output_tokens=128,
    )

    client_b = FakePleumClient()
    result_32 = run_case(
        client_b,
        {
            "id": "two",
            "task": "What must not be changed?",
            "expected_contains": ["public APIs"],
            "context": [
                {
                    "key": "a",
                    "content": "Do not change public APIs.",
                    "required": True,
                }
            ],
        },
        model="gpt-6-luna",
        token_budget=32,
        max_output_tokens=128,
    )

    grouped = summarize_by_budget([result_16, result_32])
    assert [row["token_budget"] for row in grouped] == [16, 32]


def test_markdown_total_report_contains_core_metrics() -> None:
    client = FakePleumClient()
    result = run_case(
        client,
        {
            "id": "one",
            "group": "long-task",
            "turn": 1,
            "task": "What must not be changed?",
            "expected_contains": ["public APIs"],
            "context": [
                {
                    "key": "a",
                    "content": "Do not change public APIs.",
                    "required": True,
                }
            ],
        },
        model="gpt-6-luna",
        token_budget=16,
        max_output_tokens=128,
    )
    report = {
        "provider": "PleumRouter",
        "model": "gpt-6-luna",
        "workload": "long-task",
        "summary": summarize([result]),
        "by_budget": summarize_by_budget([result]),
        "long_task_progression": long_task_progression([result]),
    }

    rendered = build_markdown_report(report)
    assert "# PleumRouter Context Optimization Report" in rendered
    assert "Input tokens" in rendered
    assert "Long-task progression" in rendered
    assert "40.00%" in rendered



def test_case_slicing_supports_train_eval_split() -> None:
    cases = [{"id": str(index)} for index in range(1, 51)]

    train = _slice_cases(cases, start=1, end=25)
    evaluate = _slice_cases(cases, start=26, end=50)

    assert len(train) == 25
    assert train[0]["id"] == "1"
    assert train[-1]["id"] == "25"
    assert len(evaluate) == 25
    assert evaluate[0]["id"] == "26"
    assert evaluate[-1]["id"] == "50"



def test_checkpoint_round_trip_supports_resume(tmp_path: Path) -> None:
    client = FakePleumClient()
    result = run_case(
        client,
        {
            "id": "resume-case",
            "task": "What must not be changed?",
            "expected_contains": ["public APIs"],
            "context": [
                {
                    "key": "a",
                    "content": "Do not change public APIs.",
                    "required": True,
                }
            ],
        },
        model="gpt-6-luna",
        token_budget=256,
        max_output_tokens=128,
    )

    path = tmp_path / "run.json.partial.jsonl"
    _append_checkpoint(
        path,
        budget=256,
        repeat_index=1,
        result=result,
    )

    restored = _load_checkpoint(path)
    key = (256, 1, "resume-case")

    assert key in restored
    assert restored[key].raw.input_tokens == 100
    assert restored[key].optimized.input_tokens == 60
    assert restored[key].actual_saved_input_tokens == 40


def test_retry_delay_uses_exponential_backoff_and_cap() -> None:
    client = PleumRouterClient(
        api_key="test",
        max_retries=6,
        retry_base_seconds=2.0,
        retry_max_seconds=30.0,
    )

    assert client._retry_delay(0, None) == 2.0
    assert client._retry_delay(1, None) == 4.0
    assert client._retry_delay(4, None) == 30.0


def test_retry_delay_prefers_retry_after_header() -> None:
    client = PleumRouterClient(
        api_key="test",
        retry_base_seconds=2.0,
        retry_max_seconds=30.0,
    )

    assert client._retry_delay(0, {"Retry-After": "7"}) == 7.0



def test_streaming_long_workload_accumulates_without_materializing_all_turns() -> None:
    iterator = iter_long_workload(
        turns=3,
        noise_blocks_per_turn=1,
        noise_chars=64,
    )

    first = next(iterator)
    second = next(iterator)
    third = next(iterator)

    assert first["id"] == "long-turn-0001"
    assert second["id"] == "long-turn-0002"
    assert third["id"] == "long-turn-0003"
    assert len(first["context"]) < len(second["context"]) < len(third["context"])
    assert third["phase"] == "long-task"
    assert "checkpoint-0003" in third["expected_contains"]
    assert "checkpoint-0001" in third["expected_not_contains"]
