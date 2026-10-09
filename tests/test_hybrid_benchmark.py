from devseat_cognitive_guard.hybrid_benchmark import (
    HybridCaseResult,
    summarize_hybrid,
    summarize_phases,
    summarize_windows,
)
from devseat_cognitive_guard.hybrid_workload import (
    build_hybrid_100_workload,
)
from devseat_cognitive_guard.pleum_benchmark import PleumRun


def _run(arm: str, tokens: int, quality: bool) -> PleumRun:
    return PleumRun(
        arm=arm,
        input_tokens=tokens,
        output_tokens=5,
        total_tokens=tokens + 5,
        latency_ms=100.0,
        cost_krw=None,
        cost_usd=None,
        output_text="ok",
        quality_pass=quality,
    )


def _result(
    turn: int,
    phase: str,
    *,
    baseline_quality: bool,
    learned_quality: bool,
) -> HybridCaseResult:
    return HybridCaseResult(
        case_id=f"hybrid-{turn:03d}",
        phase=phase,
        turn=turn,
        raw=_run("raw", 100, True),
        baseline=_run("baseline", 60, baseline_quality),
        learned=_run("learned", 50, learned_quality),
        baseline_saved_input_tokens=40,
        learned_saved_input_tokens=50,
        learned_vs_baseline_tokens=10,
        baseline_reduction_ratio=0.4,
        learned_reduction_ratio=0.5,
        baseline_omitted_keys=("a",),
        learned_omitted_keys=("a",),
        policy_features_touched=1,
        replay_probes=1,
        replay_recovered=True,
        replay_attributed_keys=("a",),
        policy_feature_count=2,
        policy_positive_features=1,
        policy_negative_features=1,
    )


def test_hybrid_workload_has_100_cases_and_four_phases() -> None:
    cases = build_hybrid_100_workload(cases=100, decoys_per_case=9)

    assert len(cases) == 100
    assert cases[0]["phase"] == "repeat"
    assert cases[25]["phase"] == "source-shift"
    assert cases[50]["phase"] == "collision"
    assert cases[75]["phase"] == "drift"
    assert cases[-1]["turn"] == 100


def test_hybrid_drift_reuses_previously_safe_archive_family() -> None:
    cases = build_hybrid_100_workload(cases=100, decoys_per_case=2)

    before = cases[0]
    drift = cases[75]

    archive_before = [
        item for item in before["context"]
        if item["key"].startswith("archive-note-")
    ]
    archive_drift = [
        item for item in drift["context"]
        if item["key"].startswith("archive-note-")
    ]

    assert len(archive_before) == 1
    assert "Historical archive note" in archive_before[0]["content"]
    assert len(archive_drift) == 1
    assert "Critical hybrid evidence" in archive_drift[0]["content"]


def test_hybrid_summary_measures_incremental_learning_gain() -> None:
    results = [
        _result(
            1,
            "repeat",
            baseline_quality=False,
            learned_quality=True,
        ),
        _result(
            2,
            "repeat",
            baseline_quality=False,
            learned_quality=True,
        ),
    ]

    summary = summarize_hybrid(results)

    assert summary["raw_input_tokens"] == 200
    assert summary["baseline_input_tokens"] == 120
    assert summary["learned_input_tokens"] == 100
    assert summary["baseline_reduction_ratio"] == 0.4
    assert summary["learned_reduction_ratio"] == 0.5
    assert summary["learned_extra_saved_vs_baseline"] == 20
    assert summary["baseline_quality_pass_rate"] == 0.0
    assert summary["learned_quality_pass_rate"] == 1.0
    assert summary["baseline_fail_learned_pass"] == 2


def test_hybrid_phase_and_window_summaries() -> None:
    results = [
        _result(
            turn,
            "repeat" if turn <= 10 else "source-shift",
            baseline_quality=False,
            learned_quality=True,
        )
        for turn in range(1, 21)
    ]

    phases = summarize_phases(results)
    windows = summarize_windows(results, window=10)

    assert len(phases) == 2
    assert phases[0]["phase"] == "repeat"
    assert phases[1]["phase"] == "source-shift"
    assert len(windows) == 2
    assert windows[0]["turn_start"] == 1
    assert windows[0]["turn_end"] == 10
    assert windows[1]["turn_start"] == 11
    assert windows[1]["turn_end"] == 20
