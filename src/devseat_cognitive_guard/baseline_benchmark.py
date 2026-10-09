from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .baseline_workload import build_baseline_1000_workload
from .long_workload import iter_complex_long_workload, iter_long_workload
from .context_models import ContextRequest
from .context_optimizer import ContextOptimizer
from .pleum_benchmark import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    PleumRouterClient,
    PleumRun,
    _build_input,
    _invoke,
    _parse_item,
    _render_context,
)
from .token_counter import ApproxTokenCounter


@dataclass(frozen=True)
class BaselineCaseResult:
    case_id: str
    phase: str
    turn: int
    baseline: PleumRun
    estimated_raw_request_tokens: int
    estimated_baseline_request_tokens: int
    estimated_saved_request_tokens: int
    estimated_reduction_ratio: float
    estimated_stage_token_savings: Mapping[str, int]
    omitted_keys: tuple[str, ...]
    duplicate_keys: tuple[str, ...]
    superseded_keys: tuple[str, ...]
    compacted_keys: tuple[str, ...]
    chunked_keys: tuple[str, ...]
    calibrated_raw: PleumRun | None = None


def run_baseline_case(
    client: PleumRouterClient,
    case: Mapping[str, Any],
    *,
    model: str,
    token_budget: int,
    max_output_tokens: int,
    selection_strategy: str,
    calibrate_raw: bool,
) -> BaselineCaseResult:
    items = tuple(_parse_item(item) for item in case["context"])
    expected = tuple(str(value) for value in case.get("expected_contains", ()))
    forbidden = tuple(
        str(value) for value in case.get("expected_not_contains", ())
    )
    task = str(case["task"])

    optimizer = ContextOptimizer()
    optimized = optimizer.optimize(
        ContextRequest(
            context=items,
            context_budget_chars=None,
            context_budget_tokens=token_budget,
            selection_strategy=selection_strategy,
        )
    )

    baseline_context = _render_context(optimized.reduced_context.items)

    baseline = _invoke(
        client,
        arm="baseline",
        model=model,
        task=task,
        context_text=baseline_context,
        max_output_tokens=max_output_tokens,
        expected=expected,
        forbidden=forbidden,
    )

    calibrated_raw = None
    if calibrate_raw:
        raw_context = _render_context(items)
        calibrated_raw = _invoke(
            client,
            arm="raw-calibration",
            model=model,
            task=task,
            context_text=raw_context,
            max_output_tokens=max_output_tokens,
            expected=expected,
            forbidden=forbidden,
        )

    counter = ApproxTokenCounter()
    estimated_raw = _estimate_request_tokens(
        task=task,
        items=items,
        counter=counter,
    )
    estimated_baseline = counter.count(
        _build_input(task=task, context_text=baseline_context)
    )
    estimated_saved = estimated_raw - estimated_baseline

    return BaselineCaseResult(
        case_id=str(case["id"]),
        phase=str(case.get("phase", case.get("group", "baseline"))),
        turn=int(case["turn"]),
        baseline=baseline,
        estimated_raw_request_tokens=estimated_raw,
        estimated_baseline_request_tokens=estimated_baseline,
        estimated_saved_request_tokens=estimated_saved,
        estimated_reduction_ratio=(
            estimated_saved / estimated_raw if estimated_raw else 0.0
        ),
        estimated_stage_token_savings=dict(optimized.stage_token_savings),
        omitted_keys=optimized.reduced_context.omitted_keys,
        duplicate_keys=optimized.reduced_context.duplicate_keys,
        superseded_keys=optimized.reduced_context.superseded_keys,
        compacted_keys=optimized.reduced_context.compacted_keys,
        chunked_keys=optimized.reduced_context.chunked_keys,
        calibrated_raw=calibrated_raw,
    )



def _estimate_request_tokens(
    *,
    task: str,
    items: Sequence,
    counter: ApproxTokenCounter,
) -> int:
    """Estimate a rendered request without joining the full raw context."""

    fixed = counter.count(_build_input(task=task, context_text=""))
    rendered_items = sum(
        counter.count(f"[{item.key}]\n{item.content}")
        for item in items
    )
    separators = max(0, len(items) - 1) * counter.count("\n\n")
    return fixed + rendered_items + separators


def summarize(
    results: Sequence[BaselineCaseResult],
) -> dict[str, Any]:
    estimated_raw = sum(
        item.estimated_raw_request_tokens for item in results
    )
    estimated_baseline = sum(
        item.estimated_baseline_request_tokens for item in results
    )
    estimated_saved = estimated_raw - estimated_baseline

    calibrated = [
        item for item in results
        if item.calibrated_raw is not None
    ]
    calibrated_raw_tokens = sum(
        item.calibrated_raw.input_tokens
        for item in calibrated
        if item.calibrated_raw is not None
    )
    calibrated_baseline_tokens = sum(
        item.baseline.input_tokens for item in calibrated
    )
    calibrated_saved = calibrated_raw_tokens - calibrated_baseline_tokens

    stage_savings: dict[str, int] = {}
    for item in results:
        for stage, value in item.estimated_stage_token_savings.items():
            stage_savings[stage] = stage_savings.get(stage, 0) + int(value)

    return {
        "cases": len(results),
        "baseline_api_calls": len(results),
        "baseline_actual_input_tokens": sum(
            item.baseline.input_tokens for item in results
        ),
        "baseline_actual_output_tokens": sum(
            item.baseline.output_tokens for item in results
        ),
        "estimated_raw_request_tokens": estimated_raw,
        "estimated_baseline_request_tokens": estimated_baseline,
        "estimated_saved_request_tokens": estimated_saved,
        "estimated_reduction_ratio": (
            estimated_saved / estimated_raw if estimated_raw else 0.0
        ),
        "quality_pass_rate": _pass_rate(
            item.baseline.quality_pass for item in results
        ),
        "quality_failures": sum(
            not item.baseline.quality_pass for item in results
        ),
        "median_latency_ms": _median(
            item.baseline.latency_ms for item in results
        ),
        "p95_latency_ms": _percentile(
            [item.baseline.latency_ms for item in results],
            0.95,
        ),
        "calibrated_cases": len(calibrated),
        "calibrated_raw_input_tokens": calibrated_raw_tokens,
        "calibrated_baseline_input_tokens": calibrated_baseline_tokens,
        "calibrated_saved_input_tokens": calibrated_saved,
        "calibrated_actual_reduction_ratio": (
            calibrated_saved / calibrated_raw_tokens
            if calibrated_raw_tokens
            else None
        ),
        "estimated_stage_token_savings": stage_savings,
        "duplicate_items_removed": sum(
            len(item.duplicate_keys) for item in results
        ),
        "superseded_items_removed": sum(
            len(item.superseded_keys) for item in results
        ),
        "compacted_items": sum(
            len(item.compacted_keys) for item in results
        ),
        "chunked_items": sum(
            len(item.chunked_keys) for item in results
        ),
        "omitted_items_or_chunks": sum(
            len(item.omitted_keys) for item in results
        ),
    }


def summarize_phases(
    results: Sequence[BaselineCaseResult],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[BaselineCaseResult]] = {}
    for item in results:
        grouped.setdefault(item.phase, []).append(item)

    return [
        {"phase": phase, **summarize(items)}
        for phase, items in grouped.items()
    ]


def summarize_windows(
    results: Sequence[BaselineCaseResult],
    *,
    window: int = 100,
) -> list[dict[str, Any]]:
    ordered = sorted(results, key=lambda item: item.turn)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(ordered), window):
        selected = ordered[start : start + window]
        if not selected:
            continue
        rows.append(
            {
                "turn_start": selected[0].turn,
                "turn_end": selected[-1].turn,
                **summarize(selected),
            }
        )
    return rows


def build_markdown_report(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    calibrated_ratio = summary["calibrated_actual_reduction_ratio"]
    calibrated_text = (
        "n/a"
        if calibrated_ratio is None
        else f"{calibrated_ratio:.2%}"
    )

    lines = [
        "# DevSeat Deterministic Baseline Report",
        "",
        f"- Model: {report['model']}",
        f"- Workload: {report.get('workload', 'independent-stress')}",
        f"- Cases/turns: {summary['cases']}",
        f"- Token budget: {report['token_budget']}",
        f"- Selection strategy: {report['selection_strategy']}",
        f"- RAW calibration interval: {report['raw_calibration_every']}",
        "",
        "## Total",
        "",
        f"- Baseline quality: {summary['quality_pass_rate']:.2%}",
        f"- Quality failures: {summary['quality_failures']}",
        f"- Baseline API calls: {summary['baseline_api_calls']}",
        f"- Actual baseline input tokens: {summary['baseline_actual_input_tokens']}",
        (
            f"- Estimated request reduction: "
            f"{summary['estimated_reduction_ratio']:.2%}"
        ),
        (
            f"- RAW-calibrated actual reduction: {calibrated_text} "
            f"({summary['calibrated_cases']} sampled cases)"
        ),
        f"- Median latency: {summary['median_latency_ms']:.1f} ms",
        f"- p95 latency: {summary['p95_latency_ms']:.1f} ms",
        "",
        "## Structural activity",
        "",
        f"- Duplicate items removed: {summary['duplicate_items_removed']}",
        f"- Superseded items removed: {summary['superseded_items_removed']}",
        f"- Compacted items: {summary['compacted_items']}",
        f"- Chunked items: {summary['chunked_items']}",
        f"- Omitted items/chunks: {summary['omitted_items_or_chunks']}",
        "",
        "## By phase",
        "",
        "| Phase | Cases | Quality | Est. reduction | Calibration reduction | Median latency |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in report["by_phase"]:
        ratio = row["calibrated_actual_reduction_ratio"]
        ratio_text = "n/a" if ratio is None else f"{ratio:.2%}"
        lines.append(
            f"| {row['phase']} | {row['cases']} | "
            f"{row['quality_pass_rate']:.2%} | "
            f"{row['estimated_reduction_ratio']:.2%} | "
            f"{ratio_text} | {row['median_latency_ms']:.1f} ms |"
        )

    lines.extend(
        [
            "",
            "## 100-case stability windows",
            "",
            "| Turns | Quality | Est. reduction | Failures | p95 latency |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["windows"]:
        lines.append(
            f"| {row['turn_start']}-{row['turn_end']} | "
            f"{row['quality_pass_rate']:.2%} | "
            f"{row['estimated_reduction_ratio']:.2%} | "
            f"{row['quality_failures']} | "
            f"{row['p95_latency_ms']:.1f} ms |"
        )

    lines.extend(
        [
            "",
            "## Estimated savings by deterministic stage",
            "",
            "| Stage | Estimated saved tokens |",
            "|---|---:|",
        ]
    )
    for stage, value in summary["estimated_stage_token_savings"].items():
        lines.append(f"| {stage} | {value} |")

    lines.extend(
        [
            "",
            "The default 1000-case mode performs Baseline calls only. "
            "RAW token savings are therefore locally estimated unless "
            "--raw-calibration-every is enabled.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="context-baseline-bench",
        description=(
            "Run a deterministic baseline-only stress benchmark. "
            "Default: 1000 baseline model calls."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--base-url",
        default=os.getenv("PLEUM_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--noise-items", type=int, default=8)
    parser.add_argument(
        "--long",
        action="store_true",
        help=(
            "Run one cumulative long task instead of independent stress cases. "
            "Each turn contains all prior state/log history."
        ),
    )
    parser.add_argument(
        "--complex-long",
        action="store_true",
        help=(
            "Run the harder cumulative multi-state workload with conflicting "
            "history and policy-derived answers."
        ),
    )
    parser.add_argument(
        "--long-turns",
        type=int,
        default=1000,
        help="Number of cumulative turns when --long is enabled.",
    )
    parser.add_argument("--long-noise-blocks", type=int, default=4)
    parser.add_argument("--long-noise-chars", type=int, default=320)
    parser.add_argument("--token-budget", type=int, default=200)
    parser.add_argument(
        "--selection-strategy",
        choices=("priority", "density", "balanced"),
        default="balanced",
    )
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--retry-max-seconds", type=float, default=30.0)
    parser.add_argument(
        "--raw-calibration-every",
        type=int,
        default=0,
        help=(
            "Also send RAW on every Nth case. 0 keeps the run baseline-only."
        ),
    )
    parser.add_argument(
        "--output",
        default="reports/baseline1000.json",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.cases < 1:
        raise SystemExit("--cases must be >= 1")
    if args.raw_calibration_every < 0:
        raise SystemExit("--raw-calibration-every must be >= 0")

    api_key = os.getenv("PLEUM_API_KEY")
    if not api_key:
        raise SystemExit("PLEUM_API_KEY is not set")

    client = PleumRouterClient(
        api_key=api_key,
        base_url=args.base_url,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        retry_max_seconds=args.retry_max_seconds,
    )
    models = client.list_models()
    if models and args.model not in models:
        raise SystemExit(
            f"model {args.model!r} was not returned by PleumRouter /v1/models"
        )

    if args.long and args.complex_long:
        raise SystemExit("--long and --complex-long are mutually exclusive")

    if args.complex_long:
        if args.long_turns < 1:
            raise SystemExit("--long-turns must be >= 1")
        cases = iter_complex_long_workload(
            turns=args.long_turns,
            noise_blocks_per_turn=args.long_noise_blocks,
            noise_chars=max(64, args.long_noise_chars),
        )
        total_cases = args.long_turns
        workload = "cumulative-complex-task"
    elif args.long:
        if args.long_turns < 1:
            raise SystemExit("--long-turns must be >= 1")
        cases = iter_long_workload(
            turns=args.long_turns,
            noise_blocks_per_turn=args.long_noise_blocks,
            noise_chars=args.long_noise_chars,
        )
        total_cases = args.long_turns
        workload = "cumulative-long-task"
    else:
        cases = build_baseline_1000_workload(
            cases=args.cases,
            noise_items=args.noise_items,
        )
        total_cases = args.cases
        workload = "independent-stress"

    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(
        output_path.suffix + ".partial.jsonl"
    )

    if args.resume:
        results = _load_checkpoint(checkpoint_path)
    else:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        results = []

    completed = {item.case_id for item in results}
    calibration_calls = (
        0
        if args.raw_calibration_every == 0
        else total_cases // args.raw_calibration_every
    )
    print(
        f"workload={workload} turns={total_cases} "
        f"baseline_api_calls={total_cases} "
        f"raw_calibration_calls≈{calibration_calls} "
        f"strategy={args.selection_strategy}"
    )

    for position, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        if case_id in completed:
            print(f"[{position}/{total_cases}] {case_id} SKIP (checkpoint)")
            continue

        calibrate = (
            args.raw_calibration_every > 0
            and position % args.raw_calibration_every == 0
        )
        result = run_baseline_case(
            client,
            case,
            model=args.model,
            token_budget=args.token_budget,
            max_output_tokens=args.max_output_tokens,
            selection_strategy=args.selection_strategy,
            calibrate_raw=calibrate,
        )
        results.append(result)
        _append_checkpoint(checkpoint_path, result)

        calibration_note = ""
        if result.calibrated_raw is not None:
            actual_saved = (
                result.calibrated_raw.input_tokens
                - result.baseline.input_tokens
            )
            calibration_note = f" actual_saved={actual_saved}"

        print(
            f"[{position}/{total_cases}] {case_id} "
            f"phase={result.phase} "
            f"quality={result.baseline.quality_pass} "
            f"input={result.baseline.input_tokens} "
            f"est_reduction={result.estimated_reduction_ratio:.2%}"
            f"{calibration_note}"
        )

    summary = summarize(results)
    report = {
        "provider": "PleumRouter",
        "model": args.model,
        "cases": total_cases,
        "workload": workload,
        "token_budget": args.token_budget,
        "selection_strategy": args.selection_strategy,
        "raw_calibration_every": args.raw_calibration_every,
        "summary": summary,
        "by_phase": summarize_phases(results),
        "windows": summarize_windows(results),
        "results": [asdict(item) for item in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(
        build_markdown_report(report),
        encoding="utf-8",
    )

    calibrated_ratio = summary["calibrated_actual_reduction_ratio"]
    calibrated_text = (
        "n/a"
        if calibrated_ratio is None
        else f"{calibrated_ratio:.2%}"
    )

    print("\n=== BASELINE 1000 TOTAL REPORT ===")
    print(
        f"quality: {summary['quality_pass_rate']:.2%} "
        f"(failures={summary['quality_failures']})"
    )
    print(
        "estimated request tokens: "
        f"{summary['estimated_raw_request_tokens']} -> "
        f"{summary['estimated_baseline_request_tokens']} "
        f"(reduction={summary['estimated_reduction_ratio']:.2%})"
    )
    print(
        f"actual baseline input tokens: "
        f"{summary['baseline_actual_input_tokens']}"
    )
    print(
        "raw calibration: "
        f"cases={summary['calibrated_cases']} "
        f"actual_reduction={calibrated_text}"
    )
    print(
        f"latency: median={summary['median_latency_ms']:.1f}ms "
        f"p95={summary['p95_latency_ms']:.1f}ms"
    )
    print(
        "structural: "
        f"duplicates={summary['duplicate_items_removed']} "
        f"superseded={summary['superseded_items_removed']} "
        f"compacted={summary['compacted_items']} "
        f"chunked={summary['chunked_items']} "
        f"omitted={summary['omitted_items_or_chunks']}"
    )
    print(f"json report: {output_path}")
    print(f"markdown report: {markdown_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return 0


def _pass_rate(values: Iterable[bool]) -> float:
    collected = [1.0 if value else 0.0 for value in values]
    return sum(collected) / len(collected) if collected else 0.0


def _median(values: Iterable[float]) -> float:
    collected = list(values)
    return statistics.median(collected) if collected else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, round((len(ordered) - 1) * quantile)),
    )
    return ordered[index]


def _append_checkpoint(
    path: Path,
    result: BaselineCaseResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def _load_checkpoint(path: Path) -> list[BaselineCaseResult]:
    if not path.exists():
        return []

    results: list[BaselineCaseResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        calibrated_raw = (
            None
            if raw.get("calibrated_raw") is None
            else PleumRun(**raw["calibrated_raw"])
        )
        results.append(
            BaselineCaseResult(
                case_id=str(raw["case_id"]),
                phase=str(raw["phase"]),
                turn=int(raw["turn"]),
                baseline=PleumRun(**raw["baseline"]),
                estimated_raw_request_tokens=int(
                    raw["estimated_raw_request_tokens"]
                ),
                estimated_baseline_request_tokens=int(
                    raw["estimated_baseline_request_tokens"]
                ),
                estimated_saved_request_tokens=int(
                    raw["estimated_saved_request_tokens"]
                ),
                estimated_reduction_ratio=float(
                    raw["estimated_reduction_ratio"]
                ),
                estimated_stage_token_savings={
                    str(key): int(value)
                    for key, value in raw[
                        "estimated_stage_token_savings"
                    ].items()
                },
                omitted_keys=tuple(raw["omitted_keys"]),
                duplicate_keys=tuple(raw["duplicate_keys"]),
                superseded_keys=tuple(raw["superseded_keys"]),
                compacted_keys=tuple(raw["compacted_keys"]),
                chunked_keys=tuple(raw["chunked_keys"]),
                calibrated_raw=calibrated_raw,
            )
        )
    return results


if __name__ == "__main__":
    raise SystemExit(main())
