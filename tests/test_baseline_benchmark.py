from devseat_cognitive_guard.baseline_benchmark import (
    BaselineCaseResult,
    _estimate_request_tokens,
    summarize,
    summarize_phases,
    summarize_windows,
)
from devseat_cognitive_guard.baseline_workload import (
    PHASES,
    build_baseline_1000_workload,
)
from devseat_cognitive_guard.long_workload import (
    build_complex_long_workload,
    iter_complex_long_workload,
)
from devseat_cognitive_guard.context_models import ContextItem
from devseat_cognitive_guard.pleum_benchmark import PleumRun
from devseat_cognitive_guard.token_counter import ApproxTokenCounter


def _run(tokens: int, quality: bool = True) -> PleumRun:
    return PleumRun(
        arm="baseline",
        input_tokens=tokens,
        output_tokens=5,
        total_tokens=tokens + 5,
        latency_ms=100.0,
        cost_krw=None,
        cost_usd=None,
        output_text="BL-0001 B1000",
        quality_pass=quality,
    )


def _result(
    turn: int,
    phase: str,
    *,
    quality: bool = True,
    calibrated: bool = False,
) -> BaselineCaseResult:
    return BaselineCaseResult(
        case_id=f"baseline-{turn:04d}",
        phase=phase,
        turn=turn,
        baseline=_run(50, quality),
        estimated_raw_request_tokens=100,
        estimated_baseline_request_tokens=50,
        estimated_saved_request_tokens=50,
        estimated_reduction_ratio=0.5,
        estimated_stage_token_savings={
            "block_dedup": 5,
            "item_dedup": 5,
            "supersession": 5,
            "delta": 0,
            "budget": 35,
        },
        omitted_keys=("noise",),
        duplicate_keys=("copy",),
        superseded_keys=("old",),
        compacted_keys=("log",),
        chunked_keys=("history",),
        calibrated_raw=(
            PleumRun(
                arm="raw-calibration",
                input_tokens=100,
                output_tokens=5,
                total_tokens=105,
                latency_ms=120.0,
                cost_krw=None,
                cost_usd=None,
                output_text="BL-0001 B1000",
                quality_pass=True,
            )
            if calibrated
            else None
        ),
    )


def test_baseline_1000_workload_has_five_200_case_phases() -> None:
    cases = build_baseline_1000_workload(cases=1000, noise_items=8)

    assert len(cases) == 1000
    assert [cases[index]["phase"] for index in (0, 200, 400, 600, 800)] == list(PHASES)
    assert cases[-1]["turn"] == 1000


def test_baseline_workload_contains_safe_chunking_and_code_whitespace_case() -> None:
    cases = build_baseline_1000_workload(cases=1000, noise_items=2)
    case = cases[800]

    chunk_items = [
        item for item in case["context"]
        if item.get("chunk_mode") == "paragraph"
    ]
    code_items = [
        item for item in case["context"]
        if item.get("source") == "code"
    ]

    assert len(chunk_items) == 1
    assert chunk_items[0]["chunk_target_tokens"] == 48
    assert len(code_items) == 2
    assert code_items[0]["content"] != code_items[1]["content"]


def test_baseline_summary_tracks_quality_and_calibration() -> None:
    results = [
        _result(1, "dedup", calibrated=True),
        _result(2, "dedup", quality=False),
    ]

    summary = summarize(results)

    assert summary["cases"] == 2
    assert summary["quality_pass_rate"] == 0.5
    assert summary["quality_failures"] == 1
    assert summary["estimated_reduction_ratio"] == 0.5
    assert summary["calibrated_cases"] == 1
    assert summary["calibrated_actual_reduction_ratio"] == 0.5
    assert summary["duplicate_items_removed"] == 2
    assert summary["chunked_items"] == 2


def test_baseline_phase_and_window_summaries() -> None:
    results = [
        _result(turn, "dedup" if turn <= 100 else "supersession")
        for turn in range(1, 201)
    ]

    phases = summarize_phases(results)
    windows = summarize_windows(results, window=100)

    assert len(phases) == 2
    assert phases[0]["phase"] == "dedup"
    assert phases[1]["phase"] == "supersession"
    assert len(windows) == 2
    assert windows[0]["turn_start"] == 1
    assert windows[0]["turn_end"] == 100
    assert windows[1]["turn_start"] == 101
    assert windows[1]["turn_end"] == 200



def test_cumulative_raw_estimate_does_not_require_rendered_context() -> None:
    items = tuple(
        ContextItem(
            key=f"log-{index}",
            content="historical payload " * 100,
        )
        for index in range(20)
    )

    estimate = _estimate_request_tokens(
        task="Report current state.",
        items=items,
        counter=ApproxTokenCounter(),
    )

    assert estimate > 0
    assert estimate > ApproxTokenCounter().count("Report current state.")


def test_complex_long_workload_tracks_multiple_current_states_and_reasoning() -> None:
    cases = build_complex_long_workload(
        turns=3,
        noise_blocks_per_turn=2,
        noise_chars=96,
    )

    assert len(cases) == 3
    assert cases[-1]["id"] == "complex-turn-0003"
    assert cases[-1]["phase"] == "complex-long-task"

    current = [
        item
        for item in cases[-1]["context"]
        if item.get("required")
        and item.get("source") == "operator-state"
    ]
    assert len(current) == 7

    expected = cases[-1]["expected_contains"]
    assert "context_store.py" in expected
    assert "decision-0003" in expected
    assert "feature/guard-003" in expected
    assert "schema-v103" in expected
    assert "stable" in expected
    assert "svc-003.internal" in expected
    assert "high" in expected
    assert "lane-s2" in expected
    assert "dual-review" in expected
    assert "4317" in expected
    assert "12s" in expected

    forbidden = cases[-1]["expected_not_contains"]
    assert "decision-0001" in forbidden
    assert "feature/guard-001" in forbidden
    assert "schema-v101" in forbidden


def test_complex_long_workload_streams_with_growing_history() -> None:
    iterator = iter_complex_long_workload(
        turns=3,
        noise_blocks_per_turn=1,
        noise_chars=96,
    )

    first = next(iterator)
    second = next(iterator)
    third = next(iterator)

    assert len(first["context"]) < len(second["context"]) < len(third["context"])
    assert third["turn"] == 3
