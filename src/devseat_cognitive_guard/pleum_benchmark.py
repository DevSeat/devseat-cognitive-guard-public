from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .context_models import ContextItem, ContextRequest
from .context_optimizer import ContextOptimizer
from .failure_replay import (
    FailureReplayRecord,
    ReplayAttribution,
    append_replay_record,
    attribute_omission_failure,
    make_replay_record,
)
from .learning_policy import AdaptiveContextPolicy
from .learning_workload import (
    build_generalization_challenge_workload,
    build_learning_challenge_workload,
)
from .long_workload import build_long_workload


DEFAULT_BASE_URL = "https://apirouter.pleum.ai/v1"
DEFAULT_MODEL = "gpt-6-luna"


@dataclass(frozen=True)
class PleumRun:
    arm: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    cost_krw: float | None
    cost_usd: float | None
    output_text: str
    quality_pass: bool


@dataclass(frozen=True)
class PleumCaseResult:
    case_id: str
    group: str
    turn: int | None
    token_budget: int | None
    selection_strategy: str
    raw: PleumRun
    optimized: PleumRun
    estimated_saved_tokens: int
    actual_saved_input_tokens: int
    actual_reduction_ratio: float
    quality_delta: int
    retained_keys: tuple[str, ...]
    omitted_keys: tuple[str, ...]
    duplicate_keys: tuple[str, ...]
    unchanged_keys: tuple[str, ...]
    estimated_stage_token_savings: Mapping[str, int]
    policy_features_touched: int = 0
    replay_probes: int = 0
    replay_recovered: bool = False
    replay_attributed_keys: tuple[str, ...] = ()


class PleumRouterClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 120.0,
        max_retries: int = 6,
        retry_base_seconds: float = 2.0,
        retry_max_seconds: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("PLEUM_API_KEY is required")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        self.retry_base_seconds = max(0.0, float(retry_base_seconds))
        self.retry_max_seconds = max(
            self.retry_base_seconds,
            float(retry_max_seconds),
        )

    def list_models(self) -> set[str]:
        request = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        payload, _ = self._send(request)
        data = payload.get("data", payload.get("models", []))
        models: set[str] = set()
        if isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    models.add(item)
                elif isinstance(item, Mapping):
                    model_id = item.get("id") or item.get("model")
                    if model_id:
                        models.add(str(model_id))
        return models

    def responses_create(
        self,
        *,
        model: str,
        task: str,
        context_text: str,
        max_output_tokens: int,
    ) -> tuple[dict[str, Any], Mapping[str, str], float]:
        body = {
            "model": model,
            "input": _build_input(task=task, context_text=context_text),
            "max_output_tokens": max_output_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        payload, headers = self._send(request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return payload, headers, latency_ms

    def _send(
        self,
        request: urllib.request.Request,
    ) -> tuple[dict[str, Any], Mapping[str, str]]:
        retryable_http = {429, 500, 502, 503, 504}
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    return payload, dict(response.headers.items())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code not in retryable_http or attempt >= self.max_retries:
                    raise RuntimeError(
                        f"PleumRouter HTTP {exc.code}: {detail}"
                    ) from exc

                delay = self._retry_delay(attempt, exc.headers)
                print(
                    f"  transient HTTP {exc.code}; retry "
                    f"{attempt + 1}/{self.max_retries} in {delay:.1f}s"
                )
                last_error = exc
                time.sleep(delay)
            except urllib.error.URLError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"PleumRouter request failed after retries: {exc}"
                    ) from exc

                delay = self._retry_delay(attempt, None)
                print(
                    f"  transient network error; retry "
                    f"{attempt + 1}/{self.max_retries} in {delay:.1f}s"
                )
                last_error = exc
                time.sleep(delay)

        raise RuntimeError(
            f"PleumRouter request failed after retries: {last_error}"
        )

    def _retry_delay(
        self,
        attempt: int,
        headers: Mapping[str, str] | None,
    ) -> float:
        if headers is not None:
            for key, value in headers.items():
                if key.casefold() == "retry-after":
                    try:
                        retry_after = float(value)
                    except (TypeError, ValueError):
                        break
                    return max(
                        0.0,
                        min(self.retry_max_seconds, retry_after),
                    )

        return min(
            self.retry_max_seconds,
            self.retry_base_seconds * (2 ** attempt),
        )


def run_case(
    client: PleumRouterClient,
    case: Mapping[str, Any],
    *,
    model: str,
    token_budget: int | None,
    max_output_tokens: int,
    selection_strategy: str = "priority",
    learning_policy: AdaptiveContextPolicy | None = None,
    update_learning_policy: bool = False,
) -> PleumCaseResult:
    items = tuple(_parse_item(item) for item in case["context"])
    raw_context = _render_context(items)

    optimizer = ContextOptimizer(learning_policy=learning_policy)
    optimized = optimizer.optimize(
        ContextRequest(
            context=items,
            context_budget_chars=None if token_budget is not None else 12000,
            context_budget_tokens=token_budget,
            selection_strategy=selection_strategy,
        )
    )
    optimized_context = _render_context(optimized.reduced_context.items)

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
    opt = _invoke(
        client,
        arm="optimized",
        model=model,
        task=task,
        context_text=optimized_context,
        max_output_tokens=max_output_tokens,
        expected=expected,
        forbidden=forbidden,
    )

    actual_saved = raw.input_tokens - opt.input_tokens
    ratio = actual_saved / raw.input_tokens if raw.input_tokens else 0.0

    policy_features_touched = 0
    if learning_policy is not None and update_learning_policy:
        policy_features_touched = learning_policy.observe(
            items=items,
            omitted_keys=optimized.reduced_context.omitted_keys,
            raw_quality_pass=raw.quality_pass,
            optimized_quality_pass=opt.quality_pass,
        )

    raw_turn = case.get("turn")
    turn = int(raw_turn) if raw_turn is not None else None

    return PleumCaseResult(
        case_id=str(case["id"]),
        group=str(case.get("group", "default")),
        turn=turn,
        token_budget=token_budget,
        selection_strategy=selection_strategy,
        raw=raw,
        optimized=opt,
        estimated_saved_tokens=optimized.saved_context_tokens,
        actual_saved_input_tokens=actual_saved,
        actual_reduction_ratio=ratio,
        quality_delta=int(opt.quality_pass) - int(raw.quality_pass),
        retained_keys=tuple(item.key for item in optimized.reduced_context.items),
        omitted_keys=optimized.reduced_context.omitted_keys,
        duplicate_keys=optimized.reduced_context.duplicate_keys,
        unchanged_keys=optimized.reduced_context.unchanged_keys,
        estimated_stage_token_savings=optimized.stage_token_savings,
        policy_features_touched=policy_features_touched,
    )


def summarize(results: Sequence[PleumCaseResult]) -> dict[str, Any]:
    raw_input = sum(result.raw.input_tokens for result in results)
    opt_input = sum(result.optimized.input_tokens for result in results)
    raw_output = sum(result.raw.output_tokens for result in results)
    opt_output = sum(result.optimized.output_tokens for result in results)
    raw_total = sum(result.raw.total_tokens for result in results)
    opt_total = sum(result.optimized.total_tokens for result in results)
    saved_input = raw_input - opt_input

    raw_cost_values = [result.raw.cost_krw for result in results]
    opt_cost_values = [result.optimized.cost_krw for result in results]
    raw_cost = _sum_optional(raw_cost_values)
    opt_cost = _sum_optional(opt_cost_values)

    quality_regressions = sum(
        1
        for result in results
        if result.raw.quality_pass and not result.optimized.quality_pass
    )
    quality_preserved = sum(
        1
        for result in results
        if result.raw.quality_pass and result.optimized.quality_pass
    )

    estimator_errors = [
        abs(result.estimated_saved_tokens - result.actual_saved_input_tokens)
        for result in results
    ]
    stage_savings: dict[str, int] = defaultdict(int)
    for result in results:
        for stage, value in result.estimated_stage_token_savings.items():
            stage_savings[stage] += int(value)

    return {
        "cases": len(results),
        "raw_input_tokens": raw_input,
        "optimized_input_tokens": opt_input,
        "saved_input_tokens": saved_input,
        "input_token_reduction_ratio": (
            saved_input / raw_input if raw_input else 0.0
        ),
        "raw_output_tokens": raw_output,
        "optimized_output_tokens": opt_output,
        "raw_total_tokens": raw_total,
        "optimized_total_tokens": opt_total,
        "saved_total_tokens": raw_total - opt_total,
        "raw_quality_pass_rate": _mean_bool(
            result.raw.quality_pass for result in results
        ),
        "optimized_quality_pass_rate": _mean_bool(
            result.optimized.quality_pass for result in results
        ),
        "quality_preserved_cases": quality_preserved,
        "quality_regressions": quality_regressions,
        "raw_cost_krw": raw_cost,
        "optimized_cost_krw": opt_cost,
        "saved_cost_krw": (
            raw_cost - opt_cost
            if raw_cost is not None and opt_cost is not None
            else None
        ),
        "raw_cost_coverage": _coverage(raw_cost_values),
        "optimized_cost_coverage": _coverage(opt_cost_values),
        "raw_median_latency_ms": _median(
            result.raw.latency_ms for result in results
        ),
        "optimized_median_latency_ms": _median(
            result.optimized.latency_ms for result in results
        ),
        "raw_p95_latency_ms": _percentile(
            [result.raw.latency_ms for result in results],
            0.95,
        ),
        "optimized_p95_latency_ms": _percentile(
            [result.optimized.latency_ms for result in results],
            0.95,
        ),
        "median_absolute_estimator_error_tokens": _median(estimator_errors),
        "estimated_stage_token_savings": dict(stage_savings),
        "policy_features_touched": sum(
            result.policy_features_touched for result in results
        ),
        "replay_probes": sum(result.replay_probes for result in results),
        "replay_recovered_cases": sum(
            result.replay_recovered for result in results
        ),
        "replay_attributed_keys": sum(
            len(result.replay_attributed_keys) for result in results
        ),
    }


def summarize_by_budget(
    results: Sequence[PleumCaseResult],
) -> list[dict[str, Any]]:
    grouped: dict[int | None, list[PleumCaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.token_budget].append(result)

    ordered = sorted(
        grouped.items(),
        key=lambda pair: (-1 if pair[0] is None else pair[0]),
    )
    return [
        {
            "token_budget": budget,
            **summarize(group_results),
        }
        for budget, group_results in ordered
    ]


def summarize_by_group(
    results: Sequence[PleumCaseResult],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[PleumCaseResult]] = defaultdict(list)
    for result in results:
        grouped[result.group].append(result)

    return [
        {
            "group": group,
            **summarize(group_results),
        }
        for group, group_results in sorted(grouped.items())
    ]


def long_task_progression(
    results: Sequence[PleumCaseResult],
) -> list[dict[str, Any]]:
    grouped: dict[
        tuple[int | None, int | None],
        list[PleumCaseResult],
    ] = defaultdict(list)

    for result in results:
        if result.group == "long-task":
            grouped[(result.token_budget, result.turn)].append(result)

    rows: list[dict[str, Any]] = []
    for (budget, turn), group_results in sorted(
        grouped.items(),
        key=lambda pair: (
            -1 if pair[0][0] is None else pair[0][0],
            -1 if pair[0][1] is None else pair[0][1],
        ),
    ):
        raw_input = sum(item.raw.input_tokens for item in group_results)
        optimized_input = sum(
            item.optimized.input_tokens for item in group_results
        )
        saved = raw_input - optimized_input
        rows.append(
            {
                "turn": turn,
                "token_budget": budget,
                "samples": len(group_results),
                "raw_input_tokens": raw_input,
                "optimized_input_tokens": optimized_input,
                "saved_input_tokens": saved,
                "reduction_ratio": saved / raw_input if raw_input else 0.0,
                "raw_quality_pass_rate": _mean_bool(
                    item.raw.quality_pass for item in group_results
                ),
                "optimized_quality_pass_rate": _mean_bool(
                    item.optimized.quality_pass for item in group_results
                ),
            }
        )
    return rows


def build_markdown_report(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# PleumRouter Context Optimization Report",
        "",
        f"- Provider: {report['provider']}",
        f"- Model: {report['model']}",
        f"- Workload: {report['workload']}",
        f"- Selection strategy: {report.get('selection_strategy', 'priority')}",
        f"- Adaptive policy: {report.get('adaptive_policy_mode', 'off')}",
        f"- Result pairs: {summary['cases']}",
        "",
        "## Total",
        "",
        "| Metric | Raw | Optimized | Delta |",
        "|---|---:|---:|---:|",
        (
            "| Input tokens | "
            f"{summary['raw_input_tokens']} | "
            f"{summary['optimized_input_tokens']} | "
            f"-{summary['saved_input_tokens']} "
            f"({summary['input_token_reduction_ratio']:.2%}) |"
        ),
        (
            "| Output tokens | "
            f"{summary['raw_output_tokens']} | "
            f"{summary['optimized_output_tokens']} | "
            f"{summary['optimized_output_tokens'] - summary['raw_output_tokens']:+d} |"
        ),
        (
            "| Total tokens | "
            f"{summary['raw_total_tokens']} | "
            f"{summary['optimized_total_tokens']} | "
            f"-{summary['saved_total_tokens']} |"
        ),
        (
            "| Quality pass rate | "
            f"{summary['raw_quality_pass_rate']:.2%} | "
            f"{summary['optimized_quality_pass_rate']:.2%} | "
            f"{summary['optimized_quality_pass_rate'] - summary['raw_quality_pass_rate']:+.2%} |"
        ),
        (
            "| Median latency | "
            f"{summary['raw_median_latency_ms']:.1f} ms | "
            f"{summary['optimized_median_latency_ms']:.1f} ms | "
            f"{summary['optimized_median_latency_ms'] - summary['raw_median_latency_ms']:+.1f} ms |"
        ),
        "",
        f"- Quality preserved: {summary['quality_preserved_cases']}",
        f"- Quality regressions: {summary['quality_regressions']}",
        f"- Failure replay probes: {summary['replay_probes']}",
        f"- Replay-recovered regressions: {summary['replay_recovered_cases']}",
        f"- Replay-attributed keys: {summary['replay_attributed_keys']}",
        "",
        "## Estimated savings by reduction stage",
        "",
        "| Stage | Estimated saved tokens |",
        "|---|---:|",
    ]

    for stage, saved in summary["estimated_stage_token_savings"].items():
        lines.append(f"| {stage} | {saved} |")

    lines.extend([
        "",
        "Stage attribution uses the local token counter; the Total table uses actual PleumRouter usage.",
        "",
        "## By token budget",
        "",
        "| Budget | Raw input | Optimized input | Saved | Reduction | Opt quality |",
        "|---:|---:|---:|---:|---:|---:|",
    ])

    for item in report["by_budget"]:
        budget = "none" if item["token_budget"] is None else str(item["token_budget"])
        lines.append(
            f"| {budget} | {item['raw_input_tokens']} | "
            f"{item['optimized_input_tokens']} | {item['saved_input_tokens']} | "
            f"{item['input_token_reduction_ratio']:.2%} | "
            f"{item['optimized_quality_pass_rate']:.2%} |"
        )

    if report["long_task_progression"]:
        lines.extend(
            [
                "",
                "## Long-task progression",
                "",
                "| Turn | Budget | Raw input | Optimized input | Saved | Reduction | Quality |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in report["long_task_progression"]:
            budget = (
                "none"
                if item["token_budget"] is None
                else str(item["token_budget"])
            )
            quality = f"{item['optimized_quality_pass_rate']:.2%}"
            lines.append(
                f"| {item['turn']} | {budget} | {item['raw_input_tokens']} | "
                f"{item['optimized_input_tokens']} | {item['saved_input_tokens']} | "
                f"{item['reduction_ratio']:.2%} | {quality} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                f"Across all measured pairs, the optimizer reduced actual "
                f"PleumRouter input tokens by "
                f"{summary['input_token_reduction_ratio']:.2%}."
            ),
            (
                f"Optimized quality pass rate was "
                f"{summary['optimized_quality_pass_rate']:.2%}, with "
                f"{summary['quality_regressions']} quality regression(s)."
            ),
            "",
            "The JSON report contains per-case outputs and diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="context-pleum-bench",
        description=(
            "Compare raw vs ContextOptimizer input tokens on PleumRouter."
        ),
    )
    parser.add_argument(
        "--cases",
        default="benchmarks/pleum_context_cases.json",
        help="Benchmark case JSON path.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--base-url",
        default=os.getenv("PLEUM_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument(
        "--selection-strategy",
        choices=("priority", "density"),
        default="priority",
    )
    parser.add_argument(
        "--budgets",
        default=None,
        help="Comma-separated token budgets, e.g. 128,256,512.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--retry-max-seconds", type=float, default=30.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed pairs from the run checkpoint file.",
    )
    parser.add_argument(
        "--policy",
        default=None,
        help="Load and apply an adaptive policy without updating it.",
    )
    parser.add_argument(
        "--learn-policy",
        default=None,
        help="Load/apply/update an adaptive policy JSON after each pair.",
    )
    parser.add_argument(
        "--failure-replay",
        action="store_true",
        help="Probe omitted context after compression regressions to attribute the failure.",
    )
    parser.add_argument(
        "--max-replay-probes",
        type=int,
        default=8,
        help="Maximum extra model calls used to attribute one regression.",
    )
    parser.add_argument(
        "--replay-dataset",
        default=None,
        help="Append metadata-only failure attribution records as JSONL.",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--long",
        action="store_true",
        help="Use generated accumulated-context long-task workload.",
    )
    parser.add_argument(
        "--learning-challenge",
        action="store_true",
        help="Use a workload designed to test whether adaptive policy learning recovers pruned evidence.",
    )
    parser.add_argument(
        "--generalization-challenge",
        action="store_true",
        help="Use a workload where the critical evidence source changes halfway through.",
    )
    parser.add_argument("--challenge-turns", type=int, default=12)
    parser.add_argument("--challenge-decoys", type=int, default=9)
    parser.add_argument("--source-switch-turn", type=int, default=7)
    parser.add_argument(
        "--long50",
        action="store_true",
        help="Shortcut for --long --long-turns 50.",
    )
    parser.add_argument("--long-turns", type=int, default=12)
    parser.add_argument("--long-noise-blocks", type=int, default=4)
    parser.add_argument("--long-noise-chars", type=int, default=320)
    parser.add_argument(
        "--case-start",
        type=int,
        default=None,
        help="1-based first case/turn to run after workload generation.",
    )
    parser.add_argument(
        "--case-end",
        type=int,
        default=None,
        help="1-based inclusive last case/turn to run.",
    )
    parser.add_argument(
        "--output",
        default="pleum-context-benchmark.json",
    )
    parser.add_argument(
        "--markdown-output",
        default=None,
        help="Markdown total report path. Defaults next to --output.",
    )
    args = parser.parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    if args.long50:
        args.long = True
        args.long_turns = 50

    budgets = _parse_budgets(args.budgets, args.token_budget)

    if args.policy and args.learn_policy:
        raise SystemExit("use either --policy or --learn-policy, not both")
    if args.failure_replay and not args.learn_policy:
        raise SystemExit("--failure-replay requires --learn-policy")

    policy_path = args.learn_policy or args.policy
    learning_policy = (
        AdaptiveContextPolicy.load(policy_path)
        if policy_path
        else None
    )
    update_learning_policy = bool(args.learn_policy)

    if update_learning_policy and len(budgets) > 1:
        print(
            "warning: --learn-policy with multiple budgets is adaptive and "
            "therefore not a frozen apples-to-apples budget sweep"
        )

    api_key = os.getenv("PLEUM_API_KEY")
    if not api_key:
        raise SystemExit(
            "PLEUM_API_KEY is not set. Put the real key in the environment, "
            "not in source files or command history."
        )

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

    if args.generalization_challenge:
        cases = build_generalization_challenge_workload(
            turns=args.challenge_turns,
            source_switch_turn=args.source_switch_turn,
            decoys_per_turn=args.challenge_decoys,
        )
        workload = "generalization-challenge"
    elif args.learning_challenge:
        cases = build_learning_challenge_workload(
            turns=args.challenge_turns,
            decoys_per_turn=args.challenge_decoys,
        )
        workload = "learning-challenge"
    elif args.long:
        cases = build_long_workload(
            turns=args.long_turns,
            noise_blocks_per_turn=args.long_noise_blocks,
            noise_chars=args.long_noise_chars,
        )
        workload = "long-task"
    else:
        cases = _read_cases(Path(args.cases))
        workload = str(args.cases)

    cases = _slice_cases(
        cases,
        start=args.case_start,
        end=args.case_end,
    )

    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(
        output_path.suffix + ".partial.jsonl"
    )
    completed: dict[tuple[int | None, int, str], PleumCaseResult] = {}
    if args.resume and checkpoint_path.exists():
        completed = _load_checkpoint(checkpoint_path)

    results: list[PleumCaseResult] = list(completed.values())
    total_pairs = len(budgets) * args.repeat * len(cases)
    total_api_calls = total_pairs * 2
    pair_index = 0

    replay_note = (
        f" + up to {args.max_replay_probes} calls per regression"
        if args.failure_replay
        else ""
    )
    print(
        f"workload={workload} cases={len(cases)} budgets={budgets} "
        f"repeat={args.repeat} pairs={total_pairs} "
        f"base_api_calls={total_api_calls}{replay_note}"
    )

    for budget in budgets:
        for repeat_index in range(args.repeat):
            for case in cases:
                pair_index += 1
                run_key = (budget, repeat_index + 1, str(case["id"]))
                if run_key in completed:
                    print(
                        f"[{pair_index}/{total_pairs}] "
                        f"budget={budget} repeat={repeat_index + 1}/{args.repeat} "
                        f"{case['id']} SKIP (checkpoint)"
                    )
                    continue

                print(
                    f"[{pair_index}/{total_pairs}] "
                    f"budget={budget} repeat={repeat_index + 1}/{args.repeat} "
                    f"{case['id']} raw -> optimized"
                )
                result = run_case(
                    client,
                    case,
                    model=args.model,
                    token_budget=budget,
                    max_output_tokens=args.max_output_tokens,
                    selection_strategy=args.selection_strategy,
                    learning_policy=learning_policy,
                    update_learning_policy=(
                        update_learning_policy and not args.failure_replay
                    ),
                )

                if (
                    args.failure_replay
                    and learning_policy is not None
                    and update_learning_policy
                ):
                    result, replay_record = _learn_with_failure_replay(
                        client=client,
                        case=case,
                        result=result,
                        model=args.model,
                        token_budget=budget,
                        max_output_tokens=args.max_output_tokens,
                        selection_strategy=args.selection_strategy,
                        learning_policy=learning_policy,
                        max_replay_probes=args.max_replay_probes,
                    )
                    if replay_record is not None and args.replay_dataset:
                        append_replay_record(
                            args.replay_dataset,
                            replay_record,
                        )

                results.append(result)
                _append_checkpoint(
                    checkpoint_path,
                    budget=budget,
                    repeat_index=repeat_index + 1,
                    result=result,
                )

                if update_learning_policy and args.learn_policy:
                    learning_policy.save(args.learn_policy)
                print(
                    "  "
                    f"raw={result.raw.input_tokens} "
                    f"optimized={result.optimized.input_tokens} "
                    f"saved={result.actual_saved_input_tokens} "
                    f"reduction={result.actual_reduction_ratio:.2%} "
                    f"quality={result.raw.quality_pass}->{result.optimized.quality_pass}"
                )

    report = {
        "provider": "PleumRouter",
        "base_url": args.base_url,
        "model": args.model,
        "workload": workload,
        "budgets": budgets,
        "repeat": args.repeat,
        "selection_strategy": args.selection_strategy,
        "adaptive_policy_mode": (
            "learn"
            if update_learning_policy
            else ("frozen" if learning_policy is not None else "off")
        ),
        "policy_path": policy_path,
        "failure_replay": args.failure_replay,
        "max_replay_probes": args.max_replay_probes,
        "replay_dataset": args.replay_dataset,
        "summary": summarize(results),
        "by_budget": summarize_by_budget(results),
        "by_group": summarize_by_group(results),
        "long_task_progression": long_task_progression(results),
        "policy_summary": (
            learning_policy.summary()
            if learning_policy is not None
            else None
        ),
        "policy_snapshot": (
            learning_policy.to_dict()
            if learning_policy is not None
            else None
        ),
        "results": [_result_to_dict(result) for result in results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    markdown_path = (
        Path(args.markdown_output)
        if args.markdown_output
        else output_path.with_suffix(".md")
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        build_markdown_report(report),
        encoding="utf-8",
    )

    summary = report["summary"]
    print("\n=== TOTAL REPORT ===")
    print(
        f"input tokens: {summary['raw_input_tokens']} -> "
        f"{summary['optimized_input_tokens']} "
        f"(saved {summary['saved_input_tokens']}, "
        f"{summary['input_token_reduction_ratio']:.2%})"
    )
    print(
        f"quality: {summary['raw_quality_pass_rate']:.2%} -> "
        f"{summary['optimized_quality_pass_rate']:.2%} "
        f"(regressions={summary['quality_regressions']})"
    )
    print(
        f"median latency: {summary['raw_median_latency_ms']:.1f}ms -> "
        f"{summary['optimized_median_latency_ms']:.1f}ms"
    )
    stage_text = ", ".join(
        f"{stage}={saved}"
        for stage, saved in summary["estimated_stage_token_savings"].items()
    )
    print(f"estimated stage savings: {stage_text}")
    print(
        "adaptive policy: "
        f"{report['adaptive_policy_mode']} "
        f"(features touched={summary['policy_features_touched']})"
    )
    if summary["replay_probes"]:
        print(
            "failure replay: "
            f"probes={summary['replay_probes']} "
            f"recovered={summary['replay_recovered_cases']} "
            f"attributed_keys={summary['replay_attributed_keys']}"
        )
    if report["policy_summary"] is not None:
        policy_summary = report["policy_summary"]
        print(
            "policy state: "
            f"features={policy_summary['feature_count']} "
            f"positive={policy_summary['positive_bias_features']} "
            f"negative={policy_summary['negative_bias_features']} "
            f"regressions={policy_summary['total_regressions']}"
        )
    print(f"json report: {output_path}")
    print(f"markdown report: {markdown_path}")
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return 0



def _learn_with_failure_replay(
    *,
    client: PleumRouterClient,
    case: Mapping[str, Any],
    result: PleumCaseResult,
    model: str,
    token_budget: int | None,
    max_output_tokens: int,
    selection_strategy: str,
    learning_policy: AdaptiveContextPolicy,
    max_replay_probes: int,
) -> tuple[PleumCaseResult, FailureReplayRecord | None]:
    items = tuple(_parse_item(item) for item in case["context"])

    # Successful compressed calls still provide safe-omission feedback.
    if not (result.raw.quality_pass and not result.optimized.quality_pass):
        touched = learning_policy.observe(
            items=items,
            omitted_keys=result.omitted_keys,
            raw_quality_pass=result.raw.quality_pass,
            optimized_quality_pass=result.optimized.quality_pass,
        )
        return replace(
            result,
            policy_features_touched=touched,
        ), None

    optimizer = ContextOptimizer(learning_policy=learning_policy)
    optimized = optimizer.optimize(
        ContextRequest(
            context=items,
            context_budget_chars=(
                None if token_budget is not None else 12000
            ),
            context_budget_tokens=token_budget,
            selection_strategy=selection_strategy,
        )
    )
    optimized_by_key = {
        item.key: item
        for item in optimized.reduced_context.items
    }
    original_by_key = {item.key: item for item in items}

    expected = tuple(
        str(value) for value in case.get("expected_contains", ())
    )
    forbidden = tuple(
        str(value) for value in case.get("expected_not_contains", ())
    )
    task = str(case["task"])

    def probe(restored_keys: tuple[str, ...]) -> bool:
        restored = set(restored_keys)
        replay_items: list[ContextItem] = []
        for item in items:
            if item.key in optimized_by_key:
                replay_items.append(optimized_by_key[item.key])
            elif item.key in restored and item.key in original_by_key:
                replay_items.append(original_by_key[item.key])

        run = _invoke(
            client,
            arm="replay",
            model=model,
            task=task,
            context_text=_render_context(replay_items),
            max_output_tokens=max_output_tokens,
            expected=expected,
            forbidden=forbidden,
        )
        return run.quality_pass

    attribution = attribute_omission_failure(
        result.omitted_keys,
        probe=probe,
        max_probes=max_replay_probes,
    )
    targeted_keys = (
        attribution.attributed_keys
        if attribution.recovered
        else None
    )
    touched = learning_policy.observe(
        items=items,
        omitted_keys=result.omitted_keys,
        raw_quality_pass=result.raw.quality_pass,
        optimized_quality_pass=result.optimized.quality_pass,
        attributed_keys=targeted_keys,
    )

    record = make_replay_record(
        case_id=result.case_id,
        token_budget=token_budget,
        selection_strategy=selection_strategy,
        attribution=attribution,
        items=items,
        omitted_keys=result.omitted_keys,
    )
    return replace(
        result,
        policy_features_touched=touched,
        replay_probes=attribution.probes,
        replay_recovered=attribution.recovered,
        replay_attributed_keys=attribution.attributed_keys,
    ), record


def _invoke(
    client: PleumRouterClient,
    *,
    arm: str,
    model: str,
    task: str,
    context_text: str,
    max_output_tokens: int,
    expected: Sequence[str],
    forbidden: Sequence[str] = (),
) -> PleumRun:
    payload, headers, latency_ms = client.responses_create(
        model=model,
        task=task,
        context_text=context_text,
        max_output_tokens=max_output_tokens,
    )
    usage = payload.get("usage") or {}
    output_text = _extract_output_text(payload)
    return PleumRun(
        arm=arm,
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        total_tokens=int(usage.get("total_tokens", 0)),
        latency_ms=latency_ms,
        cost_krw=_header_float(headers, "x-cost-krw"),
        cost_usd=_header_float(headers, "x-cost-usd"),
        output_text=output_text,
        quality_pass=_quality_pass(
            output_text,
            expected,
            forbidden=forbidden,
        ),
    )


def _build_input(*, task: str, context_text: str) -> str:
    return (
        "Use only the supplied context when answering. "
        "If the answer is absent, say UNKNOWN.\n\n"
        f"CONTEXT\n{context_text}\n\n"
        f"TASK\n{task}"
    )


def _render_context(items: Iterable[ContextItem]) -> str:
    return "\n\n".join(
        f"[{item.key}]\n{item.content}"
        for item in items
    )


def _parse_item(raw: Mapping[str, Any]) -> ContextItem:
    return ContextItem(
        key=str(raw["key"]),
        content=str(raw["content"]),
        priority=int(raw.get("priority", 50)),
        required=bool(raw.get("required", False)),
        stable=bool(raw.get("stable", False)),
        source=None if raw.get("source") is None else str(raw["source"]),
        revision=None if raw.get("revision") is None else str(raw["revision"]),
        state_slot=(
            None if raw.get("state_slot") is None else str(raw["state_slot"])
        ),
        state_version=(
            None if raw.get("state_version") is None else int(raw["state_version"])
        ),
        dedup_blocks=bool(raw.get("dedup_blocks", False)),
        chunk_mode=(
            None if raw.get("chunk_mode") is None else str(raw["chunk_mode"])
        ),
        chunk_target_tokens=(
            None
            if raw.get("chunk_target_tokens") is None
            else int(raw["chunk_target_tokens"])
        ),
    )


def _read_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("benchmark cases must be a non-empty JSON array")
    return payload


def _parse_budgets(
    budgets: str | None,
    single_budget: int | None,
) -> list[int | None]:
    if budgets is None:
        return [single_budget]

    values: list[int] = []
    for raw in budgets.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value < 1:
            raise ValueError("token budgets must be >= 1")
        values.append(value)

    if not values:
        raise ValueError("--budgets must contain at least one integer")
    return values


def _slice_cases(
    cases: Sequence[dict[str, Any]],
    *,
    start: int | None,
    end: int | None,
) -> list[dict[str, Any]]:
    if start is None and end is None:
        return list(cases)

    first = 1 if start is None else int(start)
    last = len(cases) if end is None else int(end)
    if first < 1:
        raise ValueError("--case-start must be >= 1")
    if last < first:
        raise ValueError("--case-end must be >= --case-start")
    if first > len(cases):
        raise ValueError("--case-start is beyond the workload size")

    return list(cases[first - 1 : min(last, len(cases))])

def _append_checkpoint(
    path: Path,
    *,
    budget: int | None,
    repeat_index: int,
    result: PleumCaseResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "budget": budget,
        "repeat_index": repeat_index,
        "case_id": result.case_id,
        "result": _result_to_dict(result),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_checkpoint(
    path: Path,
) -> dict[tuple[int | None, int, str], PleumCaseResult]:
    completed: dict[tuple[int | None, int, str], PleumCaseResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        budget = record.get("budget")
        repeat_index = int(record["repeat_index"])
        case_id = str(record["case_id"])
        completed[(budget, repeat_index, case_id)] = _result_from_dict(
            record["result"]
        )
    return completed


def _result_from_dict(raw: Mapping[str, Any]) -> PleumCaseResult:
    return PleumCaseResult(
        case_id=str(raw["case_id"]),
        group=str(raw.get("group", "default")),
        turn=(
            None if raw.get("turn") is None else int(raw["turn"])
        ),
        token_budget=(
            None
            if raw.get("token_budget") is None
            else int(raw["token_budget"])
        ),
        selection_strategy=str(raw.get("selection_strategy", "priority")),
        raw=PleumRun(**raw["raw"]),
        optimized=PleumRun(**raw["optimized"]),
        estimated_saved_tokens=int(raw["estimated_saved_tokens"]),
        actual_saved_input_tokens=int(raw["actual_saved_input_tokens"]),
        actual_reduction_ratio=float(raw["actual_reduction_ratio"]),
        quality_delta=int(raw["quality_delta"]),
        retained_keys=tuple(raw.get("retained_keys", ())),
        omitted_keys=tuple(raw.get("omitted_keys", ())),
        duplicate_keys=tuple(raw.get("duplicate_keys", ())),
        unchanged_keys=tuple(raw.get("unchanged_keys", ())),
        estimated_stage_token_savings={
            str(key): int(value)
            for key, value in raw.get(
                "estimated_stage_token_savings", {}
            ).items()
        },
        policy_features_touched=int(
            raw.get("policy_features_touched", 0)
        ),
        replay_probes=int(raw.get("replay_probes", 0)),
        replay_recovered=bool(raw.get("replay_recovered", False)),
        replay_attributed_keys=tuple(
            raw.get("replay_attributed_keys", ())
        ),
    )

def _extract_output_text(payload: Mapping[str, Any]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str):
        return direct

    parts: list[str] = []
    for item in payload.get("output", ()):
        if not isinstance(item, Mapping):
            continue
        for content in item.get("content", ()):
            if not isinstance(content, Mapping):
                continue
            if content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def _quality_pass(
    output_text: str,
    expected: Sequence[str],
    *,
    forbidden: Sequence[str] = (),
) -> bool:
    folded = _normalize_quality_text(output_text)
    has_expected = all(
        _quality_contains(folded, value)
        for value in expected
    )
    has_forbidden = any(
        _quality_contains(folded, value)
        for value in forbidden
    )
    return has_expected and not has_forbidden


def _quality_contains(folded_output: str, value: str) -> bool:
    """Match a normalized fragment without identifier-prefix collisions.

    Zero-padded identifiers are normalized before matching, so
    checkpoint-0001 still matches checkpoint-1. Identifier boundaries
    prevent checkpoint-1 from matching checkpoint-10 or checkpoint-100.
    """

    needle = _normalize_quality_text(value)
    if not needle:
        return True

    pattern = re.escape(needle)
    if needle[0].isalnum() or needle[0] in "_-":
        pattern = r"(?<![\w-])" + pattern
    if needle[-1].isalnum() or needle[-1] in "_-":
        pattern += r"(?![\w-])"

    return re.search(pattern, folded_output) is not None


def _normalize_quality_text(value: str) -> str:
    folded = value.casefold()

    # Models commonly shorten zero-padded identifier suffixes in prose:
    # checkpoint-0001 -> checkpoint-01 -> checkpoint-1.
    # Normalize only digit runs attached to an identifier character/hyphen;
    # standalone numbers remain untouched.
    def normalize_digits(match: re.Match[str]) -> str:
        digits = match.group(0)
        return str(int(digits)) if len(digits) > 1 and digits.startswith("0") else digits

    return re.sub(r"(?<=[\w-])\d+", normalize_digits, folded)


def _header_float(headers: Mapping[str, str], name: str) -> float | None:
    target = name.casefold()
    for key, value in headers.items():
        if key.casefold() == target:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _sum_optional(values: Iterable[float | None]) -> float | None:
    collected = list(values)
    if not collected or any(value is None for value in collected):
        return None
    return sum(value for value in collected if value is not None)


def _coverage(values: Iterable[float | None]) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    present = sum(value is not None for value in collected)
    return present / len(collected)


def _mean_bool(values: Iterable[bool]) -> float:
    collected = [1.0 if value else 0.0 for value in values]
    return sum(collected) / len(collected) if collected else 0.0


def _median(values: Iterable[float]) -> float:
    collected = list(values)
    return statistics.median(collected) if collected else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * quantile)))
    return ordered[index]


def _result_to_dict(result: PleumCaseResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "group": result.group,
        "turn": result.turn,
        "token_budget": result.token_budget,
        "selection_strategy": result.selection_strategy,
        "raw": asdict(result.raw),
        "optimized": asdict(result.optimized),
        "estimated_saved_tokens": result.estimated_saved_tokens,
        "actual_saved_input_tokens": result.actual_saved_input_tokens,
        "actual_reduction_ratio": result.actual_reduction_ratio,
        "quality_delta": result.quality_delta,
        "retained_keys": list(result.retained_keys),
        "omitted_keys": list(result.omitted_keys),
        "duplicate_keys": list(result.duplicate_keys),
        "unchanged_keys": list(result.unchanged_keys),
        "estimated_stage_token_savings": dict(
            result.estimated_stage_token_savings
        ),
        "policy_features_touched": result.policy_features_touched,
        "replay_probes": result.replay_probes,
        "replay_recovered": result.replay_recovered,
        "replay_attributed_keys": list(result.replay_attributed_keys),
    }


if __name__ == "__main__":
    raise SystemExit(main())
