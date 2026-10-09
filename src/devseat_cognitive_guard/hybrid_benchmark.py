from __future__ import annotations

import argparse
import json
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .context_models import ContextItem, ContextRequest
from .context_optimizer import ContextOptimizer
from .failure_replay import (
    append_replay_record,
    attribute_omission_failure,
    make_replay_record,
)
from .hybrid_workload import build_hybrid_100_workload
from .learning_policy import AdaptiveContextPolicy
from .pleum_benchmark import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    PleumRouterClient,
    PleumRun,
    _invoke,
    _parse_item,
    _render_context,
)


@dataclass(frozen=True)
class HybridCaseResult:
    case_id: str
    phase: str
    turn: int
    raw: PleumRun
    baseline: PleumRun
    learned: PleumRun
    baseline_saved_input_tokens: int
    learned_saved_input_tokens: int
    learned_vs_baseline_tokens: int
    baseline_reduction_ratio: float
    learned_reduction_ratio: float
    baseline_omitted_keys: tuple[str, ...]
    learned_omitted_keys: tuple[str, ...]
    policy_features_touched: int
    replay_probes: int
    replay_recovered: bool
    replay_attributed_keys: tuple[str, ...]
    policy_feature_count: int
    policy_positive_features: int
    policy_negative_features: int


def run_hybrid_case(
    client: PleumRouterClient,
    case: Mapping[str, Any],
    *,
    model: str,
    token_budget: int,
    max_output_tokens: int,
    policy: AdaptiveContextPolicy,
    failure_replay: bool,
    max_replay_probes: int,
    replay_dataset: str | None,
) -> HybridCaseResult:
    items = tuple(_parse_item(item) for item in case["context"])
    raw_context = _render_context(items)
    expected = tuple(str(value) for value in case.get("expected_contains", ()))
    forbidden = tuple(
        str(value) for value in case.get("expected_not_contains", ())
    )
    task = str(case["task"])

    raw = _invoke(
        client,
        arm="raw",
        model=model,
        task=task,
        context_text=raw_context,
        max_output_tokens=max_output_tokens,
        expected=expected,
        forbidden=forbidden,
    )

    baseline_result = ContextOptimizer().optimize(
        ContextRequest(
            context=items,
            context_budget_chars=None,
            context_budget_tokens=token_budget,
            selection_strategy="priority",
        )
    )
    baseline = _invoke(
        client,
        arm="baseline",
        model=model,
        task=task,
        context_text=_render_context(baseline_result.reduced_context.items),
        max_output_tokens=max_output_tokens,
        expected=expected,
        forbidden=forbidden,
    )

    learned_result = ContextOptimizer(
        learning_policy=policy,
    ).optimize(
        ContextRequest(
            context=items,
            context_budget_chars=None,
            context_budget_tokens=token_budget,
            selection_strategy="priority",
        )
    )
    learned = _invoke(
        client,
        arm="learned",
        model=model,
        task=task,
        context_text=_render_context(learned_result.reduced_context.items),
        max_output_tokens=max_output_tokens,
        expected=expected,
        forbidden=forbidden,
    )

    touched = 0
    replay_probes = 0
    replay_recovered = False
    replay_keys: tuple[str, ...] = ()

    if raw.quality_pass and not learned.quality_pass and failure_replay:
        optimized_by_key = {
            item.key: item
            for item in learned_result.reduced_context.items
        }
        original_by_key = {item.key: item for item in items}

        def probe(restored_keys: tuple[str, ...]) -> bool:
            restored = set(restored_keys)
            replay_items: list[ContextItem] = []
            for item in items:
                if item.key in optimized_by_key:
                    replay_items.append(optimized_by_key[item.key])
                elif item.key in restored and item.key in original_by_key:
                    replay_items.append(original_by_key[item.key])

            replay_run = _invoke(
                client,
                arm="replay",
                model=model,
                task=task,
                context_text=_render_context(replay_items),
                max_output_tokens=max_output_tokens,
                expected=expected,
                forbidden=forbidden,
            )
            return replay_run.quality_pass

        attribution = attribute_omission_failure(
            learned_result.reduced_context.omitted_keys,
            probe=probe,
            max_probes=max_replay_probes,
        )
        replay_probes = attribution.probes
        replay_recovered = attribution.recovered
        replay_keys = attribution.attributed_keys

        touched = policy.observe(
            items=items,
            omitted_keys=learned_result.reduced_context.omitted_keys,
            raw_quality_pass=raw.quality_pass,
            optimized_quality_pass=learned.quality_pass,
            attributed_keys=(
                attribution.attributed_keys
                if attribution.recovered
                else None
            ),
        )

        if replay_dataset:
            append_replay_record(
                replay_dataset,
                make_replay_record(
                    case_id=str(case["id"]),
                    token_budget=token_budget,
                    selection_strategy="priority",
                    attribution=attribution,
                    items=items,
                    omitted_keys=learned_result.reduced_context.omitted_keys,
                ),
            )
    else:
        touched = policy.observe(
            items=items,
            omitted_keys=learned_result.reduced_context.omitted_keys,
            raw_quality_pass=raw.quality_pass,
            optimized_quality_pass=learned.quality_pass,
        )

    policy_summary = policy.summary()
    raw_tokens = raw.input_tokens
    baseline_saved = raw_tokens - baseline.input_tokens
    learned_saved = raw_tokens - learned.input_tokens

    return HybridCaseResult(
        case_id=str(case["id"]),
        phase=str(case["phase"]),
        turn=int(case["turn"]),
        raw=raw,
        baseline=baseline,
        learned=learned,
        baseline_saved_input_tokens=baseline_saved,
        learned_saved_input_tokens=learned_saved,
        learned_vs_baseline_tokens=baseline.input_tokens - learned.input_tokens,
        baseline_reduction_ratio=(
            baseline_saved / raw_tokens if raw_tokens else 0.0
        ),
        learned_reduction_ratio=(
            learned_saved / raw_tokens if raw_tokens else 0.0
        ),
        baseline_omitted_keys=baseline_result.reduced_context.omitted_keys,
        learned_omitted_keys=learned_result.reduced_context.omitted_keys,
        policy_features_touched=touched,
        replay_probes=replay_probes,
        replay_recovered=replay_recovered,
        replay_attributed_keys=replay_keys,
        policy_feature_count=policy_summary["feature_count"],
        policy_positive_features=policy_summary["positive_bias_features"],
        policy_negative_features=policy_summary["negative_bias_features"],
    )


def summarize_hybrid(
    results: Sequence[HybridCaseResult],
) -> dict[str, Any]:
    raw_tokens = sum(item.raw.input_tokens for item in results)
    baseline_tokens = sum(item.baseline.input_tokens for item in results)
    learned_tokens = sum(item.learned.input_tokens for item in results)

    baseline_saved = raw_tokens - baseline_tokens
    learned_saved = raw_tokens - learned_tokens

    baseline_quality = _pass_rate(
        item.baseline.quality_pass for item in results
    )
    learned_quality = _pass_rate(
        item.learned.quality_pass for item in results
    )
    raw_quality = _pass_rate(item.raw.quality_pass for item in results)

    baseline_regressions = sum(
        item.raw.quality_pass and not item.baseline.quality_pass
        for item in results
    )
    learned_regressions = sum(
        item.raw.quality_pass and not item.learned.quality_pass
        for item in results
    )

    return {
        "cases": len(results),
        "raw_input_tokens": raw_tokens,
        "baseline_input_tokens": baseline_tokens,
        "learned_input_tokens": learned_tokens,
        "baseline_saved_input_tokens": baseline_saved,
        "learned_saved_input_tokens": learned_saved,
        "baseline_reduction_ratio": (
            baseline_saved / raw_tokens if raw_tokens else 0.0
        ),
        "learned_reduction_ratio": (
            learned_saved / raw_tokens if raw_tokens else 0.0
        ),
        "learned_extra_saved_vs_baseline": baseline_tokens - learned_tokens,
        "raw_quality_pass_rate": raw_quality,
        "baseline_quality_pass_rate": baseline_quality,
        "learned_quality_pass_rate": learned_quality,
        "baseline_regressions": baseline_regressions,
        "learned_regressions": learned_regressions,
        "baseline_fail_learned_pass": sum(
            (not item.baseline.quality_pass) and item.learned.quality_pass
            for item in results
        ),
        "baseline_pass_learned_fail": sum(
            item.baseline.quality_pass and (not item.learned.quality_pass)
            for item in results
        ),
        "replay_probes": sum(item.replay_probes for item in results),
        "replay_recovered_cases": sum(
            item.replay_recovered for item in results
        ),
        "replay_attributed_keys": sum(
            len(item.replay_attributed_keys) for item in results
        ),
        "raw_median_latency_ms": _median(
            item.raw.latency_ms for item in results
        ),
        "baseline_median_latency_ms": _median(
            item.baseline.latency_ms for item in results
        ),
        "learned_median_latency_ms": _median(
            item.learned.latency_ms for item in results
        ),
    }


def summarize_phases(
    results: Sequence[HybridCaseResult],
) -> list[dict[str, Any]]:
    phases = ("repeat", "source-shift", "collision", "drift")
    rows: list[dict[str, Any]] = []
    for phase in phases:
        selected = [item for item in results if item.phase == phase]
        if not selected:
            continue
        rows.append({"phase": phase, **summarize_hybrid(selected)})
    return rows


def summarize_windows(
    results: Sequence[HybridCaseResult],
    *,
    window: int = 10,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = sorted(results, key=lambda item: item.turn)
    for start in range(0, len(ordered), window):
        selected = ordered[start : start + window]
        if not selected:
            continue
        rows.append(
            {
                "turn_start": selected[0].turn,
                "turn_end": selected[-1].turn,
                **summarize_hybrid(selected),
            }
        )
    return rows


def build_markdown_report(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# DevSeat Hybrid 100 Benchmark",
        "",
        f"- Model: {report['model']}",
        f"- Cases: {summary['cases']}",
        f"- Token budget: {report['token_budget']}",
        f"- Failure replay: {report['failure_replay']}",
        "",
        "## Total",
        "",
        "| Metric | RAW | Baseline | Learned |",
        "|---|---:|---:|---:|",
        (
            f"| Input tokens | {summary['raw_input_tokens']} | "
            f"{summary['baseline_input_tokens']} | "
            f"{summary['learned_input_tokens']} |"
        ),
        (
            f"| Reduction vs RAW | - | "
            f"{summary['baseline_reduction_ratio']:.2%} | "
            f"{summary['learned_reduction_ratio']:.2%} |"
        ),
        (
            f"| Quality pass rate | {summary['raw_quality_pass_rate']:.2%} | "
            f"{summary['baseline_quality_pass_rate']:.2%} | "
            f"{summary['learned_quality_pass_rate']:.2%} |"
        ),
        (
            f"| Regressions | - | {summary['baseline_regressions']} | "
            f"{summary['learned_regressions']} |"
        ),
        (
            f"| Median latency | {summary['raw_median_latency_ms']:.1f} ms | "
            f"{summary['baseline_median_latency_ms']:.1f} ms | "
            f"{summary['learned_median_latency_ms']:.1f} ms |"
        ),
        "",
        (
            f"Learned saved {summary['learned_extra_saved_vs_baseline']} "
            "additional input tokens versus deterministic baseline."
        ),
        (
            f"Cases where baseline failed but learned passed: "
            f"{summary['baseline_fail_learned_pass']}."
        ),
        (
            f"Cases where baseline passed but learned failed: "
            f"{summary['baseline_pass_learned_fail']}."
        ),
        "",
        "## By phase",
        "",
        "| Phase | Cases | Baseline reduction | Learned reduction | Baseline quality | Learned quality |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in report["by_phase"]:
        lines.append(
            f"| {row['phase']} | {row['cases']} | "
            f"{row['baseline_reduction_ratio']:.2%} | "
            f"{row['learned_reduction_ratio']:.2%} | "
            f"{row['baseline_quality_pass_rate']:.2%} | "
            f"{row['learned_quality_pass_rate']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## 10-case progression",
            "",
            "| Turns | Baseline quality | Learned quality | Baseline reduction | Learned reduction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["windows"]:
        lines.append(
            f"| {row['turn_start']}-{row['turn_end']} | "
            f"{row['baseline_quality_pass_rate']:.2%} | "
            f"{row['learned_quality_pass_rate']:.2%} | "
            f"{row['baseline_reduction_ratio']:.2%} | "
            f"{row['learned_reduction_ratio']:.2%} |"
        )

    lines.extend(
        [
            "",
            "## Learning diagnostics",
            "",
            f"- Replay probes: {summary['replay_probes']}",
            f"- Replay-recovered cases: {summary['replay_recovered_cases']}",
            f"- Replay-attributed keys: {summary['replay_attributed_keys']}",
            f"- Final policy: {json.dumps(report['policy_summary'], ensure_ascii=False)}",
            "",
            "This is a synthetic curriculum benchmark. Real operator traces are still required before making production compression claims.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="context-hybrid-bench",
        description=(
            "Compare RAW, deterministic baseline, and adaptive learned "
            "context optimization over a sequential curriculum."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--base-url",
        default=os.getenv("PLEUM_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--decoys", type=int, default=9)
    parser.add_argument("--token-budget", type=int, default=200)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--retry-max-seconds", type=float, default=30.0)
    parser.add_argument(
        "--policy",
        default="reports/hybrid100-policy.json",
    )
    parser.add_argument(
        "--failure-replay",
        action="store_true",
    )
    parser.add_argument("--max-replay-probes", type=int, default=8)
    parser.add_argument(
        "--replay-dataset",
        default="reports/hybrid100-replay.jsonl",
    )
    parser.add_argument(
        "--output",
        default="reports/hybrid100.json",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.cases < 20:
        raise SystemExit("--cases must be >= 20")

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

    cases = build_hybrid_100_workload(
        cases=args.cases,
        decoys_per_case=args.decoys,
    )

    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(
        output_path.suffix + ".partial.jsonl"
    )
    policy_path = Path(args.policy)

    if args.resume:
        if not policy_path.exists():
            raise SystemExit(
                "--resume requires the saved hybrid policy file"
            )
        policy = AdaptiveContextPolicy.load(policy_path)
        results = _load_checkpoint(checkpoint_path)
    else:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        if policy_path.exists():
            policy_path.unlink()
        if args.failure_replay and args.replay_dataset:
            replay_path = Path(args.replay_dataset)
            if replay_path.exists():
                replay_path.unlink()

        policy = AdaptiveContextPolicy()
        results = []

    completed_ids = {item.case_id for item in results}

    print(
        f"hybrid cases={len(cases)} base_api_calls={len(cases) * 3} "
        f"failure_replay={args.failure_replay} "
        f"max_extra_calls_per_regression={args.max_replay_probes if args.failure_replay else 0}"
    )

    for position, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        if case_id in completed_ids:
            print(f"[{position}/{len(cases)}] {case_id} SKIP (checkpoint)")
            continue

        result = run_hybrid_case(
            client,
            case,
            model=args.model,
            token_budget=args.token_budget,
            max_output_tokens=args.max_output_tokens,
            policy=policy,
            failure_replay=args.failure_replay,
            max_replay_probes=args.max_replay_probes,
            replay_dataset=(
                args.replay_dataset if args.failure_replay else None
            ),
        )
        results.append(result)

        _append_checkpoint(checkpoint_path, result)
        policy.save(policy_path)

        print(
            f"[{position}/{len(cases)}] {case_id} phase={result.phase} "
            f"quality raw={result.raw.quality_pass} "
            f"baseline={result.baseline.quality_pass} "
            f"learned={result.learned.quality_pass} "
            f"tokens {result.raw.input_tokens} -> "
            f"{result.baseline.input_tokens} -> {result.learned.input_tokens} "
            f"policy_features={result.policy_feature_count}"
        )

    summary = summarize_hybrid(results)
    report = {
        "provider": "PleumRouter",
        "model": args.model,
        "cases": args.cases,
        "token_budget": args.token_budget,
        "failure_replay": args.failure_replay,
        "summary": summary,
        "by_phase": summarize_phases(results),
        "windows": summarize_windows(results),
        "policy_summary": policy.summary(),
        "policy_snapshot": policy.to_dict(),
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

    print("\n=== HYBRID 100 TOTAL REPORT ===")
    print(
        "input tokens: "
        f"RAW={summary['raw_input_tokens']} "
        f"BASELINE={summary['baseline_input_tokens']} "
        f"LEARNED={summary['learned_input_tokens']}"
    )
    print(
        "reduction: "
        f"baseline={summary['baseline_reduction_ratio']:.2%} "
        f"learned={summary['learned_reduction_ratio']:.2%}"
    )
    print(
        "quality: "
        f"raw={summary['raw_quality_pass_rate']:.2%} "
        f"baseline={summary['baseline_quality_pass_rate']:.2%} "
        f"learned={summary['learned_quality_pass_rate']:.2%}"
    )
    print(
        "regressions: "
        f"baseline={summary['baseline_regressions']} "
        f"learned={summary['learned_regressions']}"
    )
    print(
        "learned vs baseline: "
        f"extra_saved={summary['learned_extra_saved_vs_baseline']} "
        f"baseline_fail_learned_pass={summary['baseline_fail_learned_pass']} "
        f"baseline_pass_learned_fail={summary['baseline_pass_learned_fail']}"
    )
    print(
        "failure replay: "
        f"probes={summary['replay_probes']} "
        f"recovered={summary['replay_recovered_cases']} "
        f"attributed_keys={summary['replay_attributed_keys']}"
    )
    print(f"policy: {policy.summary()}")
    print(f"json report: {output_path}")
    print(f"markdown report: {markdown_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return 0


def _pass_rate(values) -> float:
    collected = [1.0 if value else 0.0 for value in values]
    return sum(collected) / len(collected) if collected else 0.0


def _median(values) -> float:
    collected = list(values)
    return statistics.median(collected) if collected else 0.0


def _append_checkpoint(
    path: Path,
    result: HybridCaseResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def _load_checkpoint(path: Path) -> list[HybridCaseResult]:
    if not path.exists():
        return []

    results: list[HybridCaseResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        results.append(
            HybridCaseResult(
                case_id=str(raw["case_id"]),
                phase=str(raw["phase"]),
                turn=int(raw["turn"]),
                raw=PleumRun(**raw["raw"]),
                baseline=PleumRun(**raw["baseline"]),
                learned=PleumRun(**raw["learned"]),
                baseline_saved_input_tokens=int(
                    raw["baseline_saved_input_tokens"]
                ),
                learned_saved_input_tokens=int(
                    raw["learned_saved_input_tokens"]
                ),
                learned_vs_baseline_tokens=int(
                    raw["learned_vs_baseline_tokens"]
                ),
                baseline_reduction_ratio=float(
                    raw["baseline_reduction_ratio"]
                ),
                learned_reduction_ratio=float(
                    raw["learned_reduction_ratio"]
                ),
                baseline_omitted_keys=tuple(
                    raw["baseline_omitted_keys"]
                ),
                learned_omitted_keys=tuple(
                    raw["learned_omitted_keys"]
                ),
                policy_features_touched=int(
                    raw["policy_features_touched"]
                ),
                replay_probes=int(raw["replay_probes"]),
                replay_recovered=bool(raw["replay_recovered"]),
                replay_attributed_keys=tuple(
                    raw["replay_attributed_keys"]
                ),
                policy_feature_count=int(raw["policy_feature_count"]),
                policy_positive_features=int(
                    raw["policy_positive_features"]
                ),
                policy_negative_features=int(
                    raw["policy_negative_features"]
                ),
            )
        )
    return results


if __name__ == "__main__":
    raise SystemExit(main())
