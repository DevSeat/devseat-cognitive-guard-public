from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agentic_benchmark import AgenticPleumClient
from .agentic_workload import iter_agentic_hell_workload
from .context_models import ContextRequest
from .context_optimizer import ContextOptimizer
from .execution_benchmark import (
    FINALIZER_TOOLS,
    SUBMIT_TOOL_NAME,
    ToolTrace,
    TracedEnvironment,
    _authoritative_state,
    _compress_evidence,
    _extract_tool_calls,
)
from .pleum_benchmark import DEFAULT_BASE_URL, DEFAULT_MODEL, _parse_item, _render_context
from .token_counter import ApproxTokenCounter


FINALIZER_SYSTEM_PROMPT = (
    "You are the finalization stage of an optimized release-control runtime. "
    "All tool discovery and authoritative evidence collection have already "
    "been completed by a deterministic ToolGraph. Use only the supplied "
    "validated evidence and active contract. Derive any remaining policy "
    "outcome, then call submit_release_decision exactly once. Do not call any "
    "other tool and do not add prose."
)

KNOWN_ROOT_CALLS: tuple[tuple[str, Mapping[str, Any]], ...] = (
    ("read_current_state", {"group": "code", "consistency": "linearizable"}),
    (
        "read_current_state",
        {"group": "deployment", "consistency": "linearizable"},
    ),
    (
        "read_current_state",
        {"group": "dependency", "consistency": "linearizable"},
    ),
    ("lookup_policy", {"name": "rollout", "version": "active"}),
    ("lookup_policy", {"name": "approval", "version": "active"}),
    ("lookup_policy", {"name": "release", "version": "active"}),
)


@dataclass(frozen=True)
class RuntimeCaseResult:
    case_id: str
    turn: int
    accepted: bool
    clean_pass: bool
    first_submit_accepted: bool
    model_calls: int
    tool_calls: int
    submit_attempts: int
    trap_calls: int
    tool_errors: int
    protocol_violations: int
    evidence_coverage: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_raw_context_tokens: int
    estimated_optimized_context_tokens: int
    estimated_context_reduction_ratio: float
    projected_evidence_tokens: int
    stable_prefix_fingerprint: str
    selected_reasoning_effort: str
    repair_used: bool
    repair_reasoning_effort: str | None
    tool_trace: tuple[ToolTrace, ...]
    final_submission: Mapping[str, Any] | None


def run_runtime_case(
    client: AgenticPleumClient,
    case: Mapping[str, Any],
    *,
    model: str,
    token_budget: int,
    selection_strategy: str,
    max_output_tokens: int,
    base_reasoning_effort: str,
    retry_reasoning_effort: str,
    repair_reasoning_effort: str,
    reasoning_mode: str | None,
    service_tier: str | None,
) -> RuntimeCaseResult:
    items = tuple(_parse_item(item) for item in case["context"])
    raw_context = _render_context(items)
    optimized = ContextOptimizer().optimize(
        ContextRequest(
            context=items,
            context_budget_chars=None,
            context_budget_tokens=token_budget,
            selection_strategy=selection_strategy,
        )
    )
    optimized_context = _render_context(optimized.reduced_context.items)

    env = TracedEnvironment(case)
    _execute_known_toolgraph(env)

    raw_evidence = _compress_evidence(env)
    projected_evidence = _project_evidence(raw_evidence)
    evidence_text = json.dumps(
        projected_evidence,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    contract_items = tuple(
        item
        for item in optimized.reduced_context.items
        if item.source in {"active-contract", "security-policy"}
    )
    contract_context = _render_context(contract_items)

    selected_effort = _select_reasoning_effort(
        env,
        base=base_reasoning_effort,
        retry=retry_reasoning_effort,
        repair=repair_reasoning_effort,
    )

    user_content = _build_finalizer_user_content(
        task=str(case["task"]),
        contract_context=contract_context,
        evidence_text=evidence_text,
    )

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    latency_ms = 0.0
    protocol_violations = 0
    model_calls = 0
    accepted = False
    repair_used = False

    payload, elapsed = _call_finalizer(
        client,
        model=model,
        user_content=user_content,
        max_output_tokens=max_output_tokens,
        reasoning_effort=selected_effort,
        reasoning_mode=reasoning_mode,
        service_tier=service_tier,
    )
    model_calls += 1
    latency_ms += elapsed
    usage = payload.get("usage") or {}
    input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
    output_tokens += int(
        usage.get("completion_tokens", usage.get("output_tokens", 0))
    )
    total_tokens += int(usage.get("total_tokens", 0))

    accepted, violation = _apply_finalizer_calls(env, payload)
    protocol_violations += violation

    if not accepted:
        repair_used = True
        rejection_context = _latest_submission_feedback(env)
        repair_content = (
            f"{user_content}\n\n"
            "REPAIR_CONTEXT\n"
            f"{json.dumps(rejection_context, ensure_ascii=False, separators=(',', ':'))}\n"
            "Recompute the decision from the same authoritative evidence. "
            "Do not guess missing values."
        )
        payload, elapsed = _call_finalizer(
            client,
            model=model,
            user_content=repair_content,
            max_output_tokens=max_output_tokens,
            reasoning_effort=repair_reasoning_effort,
            reasoning_mode=reasoning_mode,
            service_tier=service_tier,
        )
        model_calls += 1
        latency_ms += elapsed
        usage = payload.get("usage") or {}
        input_tokens += int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0))
        )
        output_tokens += int(
            usage.get("completion_tokens", usage.get("output_tokens", 0))
        )
        total_tokens += int(usage.get("total_tokens", 0))

        accepted, violation = _apply_finalizer_calls(env, payload)
        protocol_violations += violation

    counter = ApproxTokenCounter()
    raw_estimated = counter.count(raw_context)
    optimized_estimated = counter.count(optimized_context)
    reduction = (
        (raw_estimated - optimized_estimated) / raw_estimated
        if raw_estimated
        else 0.0
    )
    projected_tokens = counter.count(evidence_text)

    clean_pass = (
        accepted
        and env.first_submit_accepted
        and env.submit_attempts == 1
        and env.trap_calls == 0
        and protocol_violations == 0
    )

    return RuntimeCaseResult(
        case_id=str(case["id"]),
        turn=int(case["turn"]),
        accepted=accepted,
        clean_pass=clean_pass,
        first_submit_accepted=env.first_submit_accepted,
        model_calls=model_calls,
        tool_calls=len(env.tool_names),
        submit_attempts=env.submit_attempts,
        trap_calls=env.trap_calls,
        tool_errors=env.tool_errors,
        protocol_violations=protocol_violations,
        evidence_coverage=env.evidence_coverage,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        estimated_raw_context_tokens=raw_estimated,
        estimated_optimized_context_tokens=optimized_estimated,
        estimated_context_reduction_ratio=reduction,
        projected_evidence_tokens=projected_tokens,
        stable_prefix_fingerprint=_stable_prefix_fingerprint(),
        selected_reasoning_effort=selected_effort,
        repair_used=repair_used,
        repair_reasoning_effort=(
            repair_reasoning_effort if repair_used else None
        ),
        tool_trace=tuple(env.trace),
        final_submission=env.final_submission,
    )


def _execute_known_toolgraph(env: TracedEnvironment) -> None:
    for name, arguments in KNOWN_ROOT_CALLS:
        env.execute_traced(
            name,
            arguments,
            source="runtime-known-graph",
        )

    state = _authoritative_state(env)
    dependency_args = {"service": str(state.get("upstream", ""))}
    dependency = env.execute_traced(
        "inspect_dependency",
        dependency_args,
        source="runtime-known-graph",
    )
    if (
        dependency.get("status") == "transient_error"
        and dependency.get("retryable") is True
    ):
        env.execute_traced(
            "inspect_dependency",
            dependency_args,
            source="runtime-local-retry",
        )

    artifact = env.execute_traced(
        "read_artifact",
        {
            "path": str(state.get("artifact_path", "")),
            "freshness": "fresh",
        },
        source="runtime-known-graph",
    )
    shard_count = artifact.get("shard_count")
    if isinstance(shard_count, int):
        env.execute_traced(
            "calculator",
            {
                "operation": "majority_quorum",
                "value": shard_count,
            },
            source="runtime-known-graph",
        )


def _project_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    state = dict(evidence.get("state") or {})
    policies = dict(evidence.get("policies") or {})
    dependency = dict(evidence.get("dependency") or {})
    artifact = dict(evidence.get("artifact") or {})
    calculation = dict(evidence.get("calculation") or {})

    risk = state.get("risk")
    ring = state.get("ring")

    rollout_lane = None
    rollout_rules = policies.get("rollout")
    if isinstance(rollout_rules, Mapping):
        ring_rules = rollout_rules.get(ring)
        if isinstance(ring_rules, Mapping):
            rollout_lane = ring_rules.get(risk)

    approval_gate = None
    approval_rules = policies.get("approval")
    if isinstance(approval_rules, Mapping):
        approval_gate = approval_rules.get(risk)

    release_rules = policies.get("release")
    if not isinstance(release_rules, list):
        release_rules = []

    return {
        "state": {
            key: state[key]
            for key in (
                "module",
                "checkpoint",
                "branch",
                "schema",
                "ring",
                "risk",
                "region",
                "upstream",
            )
            if key in state
        },
        "dependency_status": dependency.get("health"),
        "artifact": {
            key: artifact[key]
            for key in ("build_id", "shard_count")
            if key in artifact
        },
        "calculated_quorum": calculation.get("result"),
        "policy_projection": {
            "rollout_lane": rollout_lane,
            "approval_gate": approval_gate,
            "release_rules": release_rules,
        },
        "evidence_coverage": evidence.get("evidence_coverage"),
    }


def _select_reasoning_effort(
    env: TracedEnvironment,
    *,
    base: str,
    retry: str,
    repair: str,
) -> str:
    if env.evidence_coverage < 1.0 or env.trap_calls:
        return repair

    classifications = {item.classification for item in env.trace}
    if "error" in classifications or "rejected" in classifications:
        return repair
    if "retryable" in classifications:
        return retry
    return base


def _build_finalizer_user_content(
    *,
    task: str,
    contract_context: str,
    evidence_text: str,
) -> str:
    return (
        f"TASK\n{task}\n\n"
        f"ACTIVE_CONTRACT_CONTEXT\n{contract_context}\n\n"
        f"PROJECTED_AUTHORITATIVE_EVIDENCE\n{evidence_text}"
    )


def _call_finalizer(
    client: AgenticPleumClient,
    *,
    model: str,
    user_content: str,
    max_output_tokens: int,
    reasoning_effort: str,
    reasoning_mode: str | None,
    service_tier: str | None,
) -> tuple[dict[str, Any], float]:
    payload, _headers, elapsed = client.chat_completions_create(
        model=model,
        messages=[
            {"role": "system", "content": FINALIZER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tools=FINALIZER_TOOLS,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_mode=reasoning_mode,
        service_tier=service_tier,
    )
    return payload, elapsed


def _apply_finalizer_calls(
    env: TracedEnvironment,
    payload: Mapping[str, Any],
) -> tuple[bool, int]:
    calls = _extract_tool_calls(payload)
    if calls is None:
        return False, 1

    accepted = False
    violations = 0
    submit_seen = 0

    for name, arguments in calls:
        if name != SUBMIT_TOOL_NAME:
            violations += 1
            continue
        submit_seen += 1
        if submit_seen > 1:
            violations += 1
            continue
        result = env.execute_traced(
            name,
            arguments,
            source="model-finalizer",
        )
        if result.get("status") == "accepted":
            accepted = True

    if submit_seen == 0:
        violations += 1
    return accepted, violations


def _latest_submission_feedback(
    env: TracedEnvironment,
) -> dict[str, Any]:
    for item in reversed(env.trace):
        if item.name == SUBMIT_TOOL_NAME:
            return {
                "previous_submission": dict(item.arguments),
                "classification": item.classification,
                "result": dict(item.result_summary),
            }
    return {
        "previous_submission": None,
        "classification": "missing_submit",
        "result": {"reason": "no valid submit tool call"},
    }


def _stable_prefix_fingerprint() -> str:
    encoded = json.dumps(
        {
            "system": FINALIZER_SYSTEM_PROMPT,
            "tools": FINALIZER_TOOLS,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def summarize(results: Sequence[RuntimeCaseResult]) -> dict[str, Any]:
    if not results:
        return {"cases": 0}

    efforts = Counter(item.selected_reasoning_effort for item in results)
    return {
        "cases": len(results),
        "accepted_rate": _mean_bool(item.accepted for item in results),
        "clean_pass_rate": _mean_bool(item.clean_pass for item in results),
        "first_submit_accept_rate": _mean_bool(
            item.first_submit_accepted for item in results
        ),
        "total_model_calls": sum(item.model_calls for item in results),
        "mean_model_calls_per_case": statistics.mean(
            item.model_calls for item in results
        ),
        "repair_rate": _mean_bool(item.repair_used for item in results),
        "total_tool_calls": sum(item.tool_calls for item in results),
        "mean_tool_calls_per_case": statistics.mean(
            item.tool_calls for item in results
        ),
        "total_submit_attempts": sum(item.submit_attempts for item in results),
        "trap_calls": sum(item.trap_calls for item in results),
        "tool_errors": sum(item.tool_errors for item in results),
        "protocol_violations": sum(
            item.protocol_violations for item in results
        ),
        "mean_evidence_coverage": statistics.mean(
            item.evidence_coverage for item in results
        ),
        "total_input_tokens": sum(item.input_tokens for item in results),
        "total_output_tokens": sum(item.output_tokens for item in results),
        "total_tokens": sum(item.total_tokens for item in results),
        "mean_tokens_per_case": statistics.mean(
            item.total_tokens for item in results
        ),
        "median_case_latency_ms": statistics.median(
            item.latency_ms for item in results
        ),
        "p95_case_latency_ms": _percentile(
            [item.latency_ms for item in results],
            0.95,
        ),
        "mean_context_reduction_ratio": statistics.mean(
            item.estimated_context_reduction_ratio for item in results
        ),
        "mean_projected_evidence_tokens": statistics.mean(
            item.projected_evidence_tokens for item in results
        ),
        "reasoning_effort_counts": dict(sorted(efforts.items())),
        "stable_prefix_fingerprint": results[0].stable_prefix_fingerprint,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    return "\n".join(
        [
            "# DevSeat v0.15 Runtime Optimizer Benchmark",
            "",
            f"- Model: {report['model']}",
            f"- Turns: {summary['cases']}",
            f"- Token budget: {report['token_budget']}",
            f"- Base reasoning: {report['base_reasoning_effort']}",
            f"- Retry reasoning: {report['retry_reasoning_effort']}",
            f"- Repair reasoning: {report['repair_reasoning_effort']}",
            "",
            "## Results",
            "",
            f"- Eventual accepted: {summary['accepted_rate']:.2%}",
            f"- Clean pass: {summary['clean_pass_rate']:.2%}",
            (
                f"- First-submit accepted: "
                f"{summary['first_submit_accept_rate']:.2%}"
            ),
            (
                f"- Mean model calls/case: "
                f"{summary['mean_model_calls_per_case']:.2f}"
            ),
            f"- Repair rate: {summary['repair_rate']:.2%}",
            f"- Total model tokens: {summary['total_tokens']}",
            (
                f"- Mean model tokens/case: "
                f"{summary['mean_tokens_per_case']:.1f}"
            ),
            (
                f"- Median case latency: "
                f"{summary['median_case_latency_ms']:.1f} ms"
            ),
            (
                f"- p95 case latency: "
                f"{summary['p95_case_latency_ms']:.1f} ms"
            ),
            (
                f"- Mean projected evidence: "
                f"{summary['mean_projected_evidence_tokens']:.1f} tokens"
            ),
            (
                f"- Context reduction: "
                f"{summary['mean_context_reduction_ratio']:.2%}"
            ),
            (
                f"- Reasoning efforts: "
                f"{json.dumps(summary['reasoning_effort_counts'], sort_keys=True)}"
            ),
            (
                f"- Stable prefix fingerprint: "
                f"{summary['stable_prefix_fingerprint']}"
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="context-runtime-bench",
        description=(
            "Benchmark v0.15 known-ToolGraph fast path, projected evidence, "
            "stable finalizer schema, and adaptive reasoning escalation."
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--base-url",
        default=os.getenv("PLEUM_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--turns", type=int, default=50)
    parser.add_argument("--noise-blocks", type=int, default=8)
    parser.add_argument("--noise-chars", type=int, default=520)
    parser.add_argument("--token-budget", type=int, default=500)
    parser.add_argument(
        "--selection-strategy",
        choices=("priority", "density", "balanced"),
        default="balanced",
    )
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--retry-max-seconds", type=float, default=30.0)
    parser.add_argument(
        "--base-reasoning-effort",
        default="medium",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument(
        "--retry-reasoning-effort",
        default="high",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument(
        "--repair-reasoning-effort",
        default="xhigh",
        choices=("none", "minimal", "low", "medium", "high", "xhigh", "max"),
    )
    parser.add_argument(
        "--reasoning-mode",
        default="pro",
        choices=("standard", "pro"),
    )
    parser.add_argument(
        "--service-tier",
        default=None,
        choices=("auto", "default", "flex", "background", "priority"),
    )
    parser.add_argument(
        "--output",
        default="reports/runtime-v015-50.json",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.turns < 1:
        raise SystemExit("--turns must be >= 1")

    api_key = os.getenv("PLEUM_API_KEY")
    if not api_key:
        raise SystemExit("PLEUM_API_KEY is not set")

    client = AgenticPleumClient(
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

    output_path = Path(args.output)
    checkpoint_path = output_path.with_suffix(
        output_path.suffix + ".partial.jsonl"
    )
    results = _load_checkpoint(checkpoint_path) if args.resume else []
    if not args.resume and checkpoint_path.exists():
        checkpoint_path.unlink()

    completed = {item.case_id for item in results}
    cases = iter_agentic_hell_workload(
        turns=args.turns,
        noise_blocks_per_turn=args.noise_blocks,
        noise_chars=args.noise_chars,
    )

    print(
        f"workload=runtime-v015 turns={args.turns} model={args.model} "
        f"reasoning={args.base_reasoning_effort}->"
        f"{args.retry_reasoning_effort}->"
        f"{args.repair_reasoning_effort}/{args.reasoning_mode} "
        f"token_budget={args.token_budget}"
    )

    for position, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        if case_id in completed:
            print(f"[{position}/{args.turns}] {case_id} SKIP (checkpoint)")
            continue

        result = run_runtime_case(
            client,
            case,
            model=args.model,
            token_budget=args.token_budget,
            selection_strategy=args.selection_strategy,
            max_output_tokens=args.max_output_tokens,
            base_reasoning_effort=args.base_reasoning_effort,
            retry_reasoning_effort=args.retry_reasoning_effort,
            repair_reasoning_effort=args.repair_reasoning_effort,
            reasoning_mode=args.reasoning_mode,
            service_tier=args.service_tier,
        )
        results.append(result)
        _append_checkpoint(checkpoint_path, result)

        print(
            f"[{position}/{args.turns}] {case_id} "
            f"accepted={result.accepted} clean={result.clean_pass} "
            f"calls={result.model_calls} tools={result.tool_calls} "
            f"effort={result.selected_reasoning_effort} "
            f"repair={result.repair_used} "
            f"tokens={result.total_tokens} "
            f"latency={result.latency_ms:.1f}ms "
            f"projected={result.projected_evidence_tokens}"
        )

    summary = summarize(results)
    report = {
        "provider": "PleumRouter",
        "model": args.model,
        "workload": "runtime-v015",
        "turns": args.turns,
        "token_budget": args.token_budget,
        "selection_strategy": args.selection_strategy,
        "base_reasoning_effort": args.base_reasoning_effort,
        "retry_reasoning_effort": args.retry_reasoning_effort,
        "repair_reasoning_effort": args.repair_reasoning_effort,
        "reasoning_mode": args.reasoning_mode,
        "service_tier": args.service_tier,
        "summary": summary,
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

    print("\n=== RUNTIME V0.15 REPORT ===")
    print(
        f"accepted={summary['accepted_rate']:.2%} "
        f"clean={summary['clean_pass_rate']:.2%} "
        f"first_submit={summary['first_submit_accept_rate']:.2%}"
    )
    print(
        f"model_calls={summary['total_model_calls']} "
        f"mean_calls={summary['mean_model_calls_per_case']:.2f} "
        f"repair_rate={summary['repair_rate']:.2%}"
    )
    print(
        f"tokens={summary['total_tokens']} "
        f"mean_tokens={summary['mean_tokens_per_case']:.1f} "
        f"median_latency={summary['median_case_latency_ms']:.1f}ms "
        f"p95={summary['p95_case_latency_ms']:.1f}ms"
    )
    print(
        f"projected_evidence={summary['mean_projected_evidence_tokens']:.1f} "
        f"context_reduction={summary['mean_context_reduction_ratio']:.2%} "
        f"efforts={summary['reasoning_effort_counts']}"
    )
    print(f"json report: {output_path}")
    print(f"markdown report: {markdown_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return 0


def _append_checkpoint(path: Path, result: RuntimeCaseResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def _load_checkpoint(path: Path) -> list[RuntimeCaseResult]:
    if not path.exists():
        return []

    results: list[RuntimeCaseResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        traces = tuple(
            ToolTrace(
                source=str(item["source"]),
                name=str(item["name"]),
                arguments=dict(item["arguments"]),
                classification=str(item["classification"]),
                result_summary=dict(item["result_summary"]),
            )
            for item in raw.get("tool_trace", ())
        )
        results.append(
            RuntimeCaseResult(
                case_id=str(raw["case_id"]),
                turn=int(raw["turn"]),
                accepted=bool(raw["accepted"]),
                clean_pass=bool(raw["clean_pass"]),
                first_submit_accepted=bool(raw["first_submit_accepted"]),
                model_calls=int(raw["model_calls"]),
                tool_calls=int(raw["tool_calls"]),
                submit_attempts=int(raw["submit_attempts"]),
                trap_calls=int(raw["trap_calls"]),
                tool_errors=int(raw["tool_errors"]),
                protocol_violations=int(raw["protocol_violations"]),
                evidence_coverage=float(raw["evidence_coverage"]),
                input_tokens=int(raw["input_tokens"]),
                output_tokens=int(raw["output_tokens"]),
                total_tokens=int(raw["total_tokens"]),
                latency_ms=float(raw["latency_ms"]),
                estimated_raw_context_tokens=int(
                    raw["estimated_raw_context_tokens"]
                ),
                estimated_optimized_context_tokens=int(
                    raw["estimated_optimized_context_tokens"]
                ),
                estimated_context_reduction_ratio=float(
                    raw["estimated_context_reduction_ratio"]
                ),
                projected_evidence_tokens=int(raw["projected_evidence_tokens"]),
                stable_prefix_fingerprint=str(
                    raw["stable_prefix_fingerprint"]
                ),
                selected_reasoning_effort=str(
                    raw["selected_reasoning_effort"]
                ),
                repair_used=bool(raw["repair_used"]),
                repair_reasoning_effort=raw.get("repair_reasoning_effort"),
                tool_trace=traces,
                final_submission=raw.get("final_submission"),
            )
        )
    return results


def _mean_bool(values: Sequence[bool] | Any) -> float:
    collected = [1.0 if value else 0.0 for value in values]
    return sum(collected) / len(collected) if collected else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(
        0,
        min(len(ordered) - 1, round((len(ordered) - 1) * quantile)),
    )
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())
