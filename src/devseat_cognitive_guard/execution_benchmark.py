from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .agentic_benchmark import (
    AgenticPleumClient,
    AgenticToolEnvironment,
    TOOL_DEFINITIONS,
)
from .agentic_workload import iter_agentic_hell_workload
from .context_models import ContextRequest
from .context_optimizer import ContextOptimizer
from .pleum_benchmark import DEFAULT_BASE_URL, DEFAULT_MODEL, PleumRouterClient, _parse_item, _render_context
from .token_counter import ApproxTokenCounter


ROOT_TOOL_NAMES = frozenset({"read_current_state", "lookup_policy"})
SUBMIT_TOOL_NAME = "submit_release_decision"
TRACE_RESULT_KEYS = (
    "status",
    "authoritative",
    "consistency",
    "version",
    "error",
    "reason",
    "retryable",
    "mismatched_fields",
    "missing",
)


@dataclass(frozen=True)
class ToolTrace:
    source: str
    name: str
    arguments: Mapping[str, Any]
    classification: str
    result_summary: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionCaseResult:
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
    compressed_evidence_tokens: int
    tool_schema_tokens_planner: int
    tool_schema_tokens_finalizer: int
    tool_trace: tuple[ToolTrace, ...]
    final_submission: Mapping[str, Any] | None


def _select_tools(names: set[str] | frozenset[str]) -> tuple[dict[str, Any], ...]:
    return tuple(
        tool
        for tool in TOOL_DEFINITIONS
        if str(tool.get("function", {}).get("name", "")) in names
    )


PLANNER_TOOLS = _select_tools(ROOT_TOOL_NAMES)
FINALIZER_TOOLS = _select_tools(frozenset({SUBMIT_TOOL_NAME}))


class TracedEnvironment(AgenticToolEnvironment):
    def __init__(self, case: Mapping[str, Any]) -> None:
        super().__init__(case)
        self.trace: list[ToolTrace] = []
        self.outputs: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []

    def execute_traced(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        source: str,
    ) -> dict[str, Any]:
        before_traps = self.trap_calls
        before_errors = self.tool_errors
        result = self.execute(name, arguments)
        classification = _classify_result(
            result,
            trap_delta=self.trap_calls - before_traps,
            error_delta=self.tool_errors - before_errors,
        )
        self.trace.append(
            ToolTrace(
                source=source,
                name=name,
                arguments=dict(arguments),
                classification=classification,
                result_summary=_result_summary(result),
            )
        )
        self.outputs.append((name, dict(arguments), dict(result)))
        return result


def run_execution_case(
    client: AgenticPleumClient,
    case: Mapping[str, Any],
    *,
    model: str,
    token_budget: int,
    selection_strategy: str,
    max_output_tokens: int,
    reasoning_effort: str | None,
    reasoning_mode: str | None,
    service_tier: str | None,
) -> ExecutionCaseResult:
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
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    latency_ms = 0.0
    protocol_violations = 0
    model_calls = 0

    planner_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are the planning stage of an optimized release-control "
                "runtime. In ONE parallel tool batch, request all authoritative "
                "root evidence needed for the task. Call read_current_state for "
                "code, deployment, and dependency with consistency=linearizable, "
                "and lookup_policy for rollout, approval, and release with "
                "version=active. Do not guess and do not use stale sources."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{case['task']}\n\n"
                "Reduced workspace context:\n"
                f"{optimized_context}"
            ),
        },
    ]

    payload, _headers, elapsed = client.chat_completions_create(
        model=model,
        messages=planner_messages,
        tools=PLANNER_TOOLS,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_mode=reasoning_mode,
        service_tier=service_tier,
    )
    model_calls += 1
    latency_ms += elapsed
    usage = payload.get("usage") or {}
    input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
    output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
    total_tokens += int(usage.get("total_tokens", 0))

    requested = _extract_tool_calls(payload)
    if requested is None:
        protocol_violations += 1
        requested = []

    seen_roots: set[tuple[str, str]] = set()
    for name, arguments in requested:
        if name not in ROOT_TOOL_NAMES:
            protocol_violations += 1
            continue
        root_result = _execute_root(
            env,
            name,
            arguments,
            source="model-planner",
        )
        marker = _root_marker(name, arguments)
        if marker is not None and _root_result_is_authoritative(
            name,
            root_result,
        ):
            seen_roots.add(marker)

    # Deterministic completion of the root plan. This is the execution optimizer:
    # missing mandatory evidence does not require another model round-trip.
    mandatory_roots = (
        ("read_current_state", {"group": "code", "consistency": "linearizable"}),
        ("read_current_state", {"group": "deployment", "consistency": "linearizable"}),
        ("read_current_state", {"group": "dependency", "consistency": "linearizable"}),
        ("lookup_policy", {"name": "rollout", "version": "active"}),
        ("lookup_policy", {"name": "approval", "version": "active"}),
        ("lookup_policy", {"name": "release", "version": "active"}),
    )
    for name, arguments in mandatory_roots:
        marker = _root_marker(name, arguments)
        if marker not in seen_roots:
            _execute_root(env, name, arguments, source="optimizer-fill")

    # Dependency expansion/coalescing: derive follow-up tool arguments from
    # authoritative structured state, execute them without another model call,
    # and retry retryable failures locally.
    authoritative_state = _authoritative_state(env)
    dependency_args = {
        "service": str(authoritative_state.get("upstream", ""))
    }
    dependency = env.execute_traced(
        "inspect_dependency",
        dependency_args,
        source="optimizer-chain",
    )
    if dependency.get("status") == "transient_error" and dependency.get("retryable"):
        dependency = env.execute_traced(
            "inspect_dependency",
            dependency_args,
            source="optimizer-retry",
        )

    artifact_args = {
        "path": str(authoritative_state.get("artifact_path", "")),
        "freshness": "fresh",
    }
    artifact = env.execute_traced(
        "read_artifact",
        artifact_args,
        source="optimizer-chain",
    )

    shard_count = artifact.get("shard_count")
    if isinstance(shard_count, int):
        env.execute_traced(
            "calculator",
            {"operation": "majority_quorum", "value": shard_count},
            source="optimizer-chain",
        )

    evidence = _compress_evidence(env)
    evidence_text = json.dumps(
        evidence,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    contract_items = tuple(
        item
        for item in optimized.reduced_context.items
        if item.source in {"active-contract", "security-policy"}
    )
    contract_context = _render_context(contract_items)

    finalizer_messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are the finalization stage of an optimized release-control "
                "runtime. The supplied evidence is authoritative and already "
                "validated. Derive the requested policy outcomes and submit the "
                "complete decision with submit_release_decision. Do not call any "
                "other tool and do not add prose."
            ),
        },
        {
            "role": "user",
            "content": (
                f"TASK\n{case['task']}\n\n"
                f"ACTIVE_CONTRACT_CONTEXT\n{contract_context}\n\n"
                f"AUTHORITATIVE_TOOL_EVIDENCE\n{evidence_text}"
            ),
        },
    ]

    payload, _headers, elapsed = client.chat_completions_create(
        model=model,
        messages=finalizer_messages,
        tools=FINALIZER_TOOLS,
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
        reasoning_mode=reasoning_mode,
        service_tier=service_tier,
    )
    model_calls += 1
    latency_ms += elapsed
    usage = payload.get("usage") or {}
    input_tokens += int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
    output_tokens += int(usage.get("completion_tokens", usage.get("output_tokens", 0)))
    total_tokens += int(usage.get("total_tokens", 0))

    final_calls = _extract_tool_calls(payload)
    accepted = False
    if final_calls is None:
        protocol_violations += 1
        final_calls = []

    for name, arguments in final_calls:
        if name != SUBMIT_TOOL_NAME:
            protocol_violations += 1
            continue
        result = env.execute_traced(
            name,
            arguments,
            source="model-finalizer",
        )
        if result.get("status") == "accepted":
            accepted = True

    counter = ApproxTokenCounter()
    raw_estimated = counter.count(raw_context)
    optimized_estimated = counter.count(optimized_context)
    reduction = (
        (raw_estimated - optimized_estimated) / raw_estimated
        if raw_estimated
        else 0.0
    )
    compressed_evidence_tokens = counter.count(evidence_text)
    planner_schema_tokens = counter.count(
        json.dumps(PLANNER_TOOLS, ensure_ascii=False, separators=(",", ":"))
    )
    finalizer_schema_tokens = counter.count(
        json.dumps(FINALIZER_TOOLS, ensure_ascii=False, separators=(",", ":"))
    )
    clean_pass = (
        accepted
        and env.first_submit_accepted
        and env.submit_attempts == 1
        and env.trap_calls == 0
        and protocol_violations == 0
    )

    return ExecutionCaseResult(
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
        compressed_evidence_tokens=compressed_evidence_tokens,
        tool_schema_tokens_planner=planner_schema_tokens,
        tool_schema_tokens_finalizer=finalizer_schema_tokens,
        tool_trace=tuple(env.trace),
        final_submission=env.final_submission,
    )


def _execute_root(
    env: TracedEnvironment,
    name: str,
    arguments: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    return env.execute_traced(name, arguments, source=source)


def _root_marker(
    name: str,
    arguments: Mapping[str, Any],
) -> tuple[str, str] | None:
    if name == "read_current_state":
        return (name, str(arguments.get("group", "")))
    if name == "lookup_policy":
        return (name, str(arguments.get("name", "")))
    return None


def _root_result_is_authoritative(
    name: str,
    result: Mapping[str, Any],
) -> bool:
    if name == "read_current_state":
        return (
            result.get("status") == "ok"
            and result.get("authoritative") is True
            and result.get("consistency") == "linearizable"
        )
    if name == "lookup_policy":
        return (
            result.get("status") == "ok"
            and result.get("version") == "active"
        )
    return False


def _extract_tool_calls(
    payload: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]] | None:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], Mapping):
        return None
    message = choices[0].get("message") or {}
    if not isinstance(message, Mapping):
        return None
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list) or not raw_calls:
        return None

    calls: list[tuple[str, Mapping[str, Any]]] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            continue
        function = raw_call.get("function") or {}
        if not isinstance(function, Mapping):
            continue
        name = str(function.get("name", ""))
        raw_arguments = function.get("arguments", "{}")
        try:
            arguments = (
                json.loads(raw_arguments)
                if isinstance(raw_arguments, str)
                else dict(raw_arguments)
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            arguments = {}
        if not isinstance(arguments, Mapping):
            arguments = {}
        calls.append((name, arguments))
    return calls


def _authoritative_state(
    env: TracedEnvironment,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name, arguments, result in env.outputs:
        if name != "read_current_state":
            continue
        if not _root_result_is_authoritative(name, result):
            continue
        group = str(arguments.get("group", ""))
        if group == "code":
            keys = ("module", "branch", "schema")
        elif group == "deployment":
            keys = ("checkpoint", "ring", "risk")
        elif group == "dependency":
            keys = ("upstream", "region", "artifact_path")
        else:
            continue
        for key in keys:
            if key in result:
                state[key] = result[key]
    return state


def _compress_evidence(
    env: TracedEnvironment,
) -> dict[str, Any]:
    state: dict[str, Any] = {}
    policies: dict[str, Any] = {}
    dependency: dict[str, Any] = {}
    artifact: dict[str, Any] = {}
    calculation: dict[str, Any] = {}

    for name, arguments, result in env.outputs:
        if name == "read_current_state" and result.get("authoritative") is True:
            group = str(arguments.get("group", ""))
            if group == "code":
                for key in ("module", "branch", "schema"):
                    if key in result:
                        state[key] = result[key]
            elif group == "deployment":
                for key in ("checkpoint", "ring", "risk"):
                    if key in result:
                        state[key] = result[key]
            elif group == "dependency":
                for key in ("upstream", "region", "artifact_path"):
                    if key in result:
                        state[key] = result[key]

        elif (
            name == "lookup_policy"
            and result.get("version") == "active"
            and result.get("status") == "ok"
        ):
            policy_name = str(arguments.get("name", ""))
            if policy_name:
                policies[policy_name] = result.get("rules")

        elif name == "inspect_dependency" and result.get("status") == "ok":
            dependency = {
                key: result[key]
                for key in ("service", "health", "latency_ms")
                if key in result
            }

        elif (
            name == "read_artifact"
            and result.get("authoritative") is True
            and result.get("freshness") == "fresh"
        ):
            artifact = {
                key: result[key]
                for key in (
                    "path",
                    "build_id",
                    "shard_count",
                    "content_digest",
                )
                if key in result
            }

        elif name == "calculator" and result.get("status") == "ok":
            calculation = {
                key: result[key]
                for key in ("operation", "input", "result")
                if key in result
            }

    return {
        "state": state,
        "policies": policies,
        "dependency": dependency,
        "artifact": artifact,
        "calculation": calculation,
        "evidence_coverage": env.evidence_coverage,
    }


def _classify_result(
    result: Mapping[str, Any],
    *,
    trap_delta: int,
    error_delta: int,
) -> str:
    if trap_delta:
        return "trap"
    status = str(result.get("status", ""))
    if status == "transient_error" and result.get("retryable"):
        return "retryable"
    if status == "rejected":
        return "rejected"
    if error_delta or status == "error":
        return "error"
    if status == "accepted":
        return "accepted"
    return "success"


def _result_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        key: result[key]
        for key in TRACE_RESULT_KEYS
        if key in result
    }
    for key in (
        "module",
        "checkpoint",
        "branch",
        "schema",
        "ring",
        "risk",
        "region",
        "upstream",
        "health",
        "build_id",
        "shard_count",
        "result",
    ):
        if key in result:
            summary[key] = result[key]
    return summary


def summarize(results: Sequence[ExecutionCaseResult]) -> dict[str, Any]:
    if not results:
        return {"cases": 0}

    return {
        "cases": len(results),
        "accepted_rate": _mean_bool(item.accepted for item in results),
        "clean_pass_rate": _mean_bool(item.clean_pass for item in results),
        "first_submit_accept_rate": _mean_bool(
            item.first_submit_accepted for item in results
        ),
        "total_model_calls": sum(item.model_calls for item in results),
        "total_tool_calls": sum(item.tool_calls for item in results),
        "total_submit_attempts": sum(item.submit_attempts for item in results),
        "trap_calls": sum(item.trap_calls for item in results),
        "tool_errors": sum(item.tool_errors for item in results),
        "protocol_violations": sum(item.protocol_violations for item in results),
        "mean_evidence_coverage": statistics.mean(
            item.evidence_coverage for item in results
        ),
        "total_input_tokens": sum(item.input_tokens for item in results),
        "total_output_tokens": sum(item.output_tokens for item in results),
        "total_tokens": sum(item.total_tokens for item in results),
        "median_case_latency_ms": statistics.median(
            item.latency_ms for item in results
        ),
        "p95_case_latency_ms": _percentile(
            [item.latency_ms for item in results], 0.95
        ),
        "mean_model_calls_per_case": statistics.mean(
            item.model_calls for item in results
        ),
        "mean_tool_calls_per_case": statistics.mean(
            item.tool_calls for item in results
        ),
        "mean_context_reduction_ratio": statistics.mean(
            item.estimated_context_reduction_ratio for item in results
        ),
        "mean_compressed_evidence_tokens": statistics.mean(
            item.compressed_evidence_tokens for item in results
        ),
        "planner_tool_schema_tokens": results[0].tool_schema_tokens_planner,
        "finalizer_tool_schema_tokens": results[0].tool_schema_tokens_finalizer,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="context-execution-bench",
        description=(
            "Benchmark DevSeat execution optimization: tool selection, "
            "call coalescing, local retry, result compression, and two-call flow."
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
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--retry-base-seconds", type=float, default=2.0)
    parser.add_argument("--retry-max-seconds", type=float, default=30.0)
    parser.add_argument(
        "--reasoning-effort",
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
        default="reports/execution-optimized50.json",
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
        f"workload=execution-optimized turns={args.turns} model={args.model} "
        f"reasoning={args.reasoning_effort}/{args.reasoning_mode} "
        f"token_budget={args.token_budget}"
    )

    for position, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        if case_id in completed:
            print(f"[{position}/{args.turns}] {case_id} SKIP (checkpoint)")
            continue

        result = run_execution_case(
            client,
            case,
            model=args.model,
            token_budget=args.token_budget,
            selection_strategy=args.selection_strategy,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            reasoning_mode=args.reasoning_mode,
            service_tier=args.service_tier,
        )
        results.append(result)
        _append_checkpoint(checkpoint_path, result)
        print(
            f"[{position}/{args.turns}] {case_id} "
            f"accepted={result.accepted} clean={result.clean_pass} "
            f"calls={result.model_calls} tools={result.tool_calls} "
            f"errors={result.tool_errors} protocol={result.protocol_violations} "
            f"tokens={result.total_tokens} latency={result.latency_ms:.1f}ms "
            f"evidence_tokens={result.compressed_evidence_tokens}"
        )

    summary = summarize(results)
    report = {
        "provider": "PleumRouter",
        "model": args.model,
        "workload": "execution-optimized",
        "turns": args.turns,
        "token_budget": args.token_budget,
        "selection_strategy": args.selection_strategy,
        "reasoning_effort": args.reasoning_effort,
        "reasoning_mode": args.reasoning_mode,
        "summary": summary,
        "results": [asdict(item) for item in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n=== EXECUTION OPTIMIZER REPORT ===")
    print(
        f"accepted={summary['accepted_rate']:.2%} "
        f"clean={summary['clean_pass_rate']:.2%} "
        f"first_submit={summary['first_submit_accept_rate']:.2%}"
    )
    print(
        f"model_calls={summary['total_model_calls']} "
        f"mean_calls={summary['mean_model_calls_per_case']:.2f} "
        f"tool_calls={summary['total_tool_calls']}"
    )
    print(
        f"tokens={summary['total_tokens']} "
        f"median_latency={summary['median_case_latency_ms']:.1f}ms "
        f"p95={summary['p95_case_latency_ms']:.1f}ms"
    )
    print(
        f"mean_evidence_tokens={summary['mean_compressed_evidence_tokens']:.1f} "
        f"context_reduction={summary['mean_context_reduction_ratio']:.2%}"
    )
    print(f"json report: {output_path}")

    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return 0


def _append_checkpoint(path: Path, result: ExecutionCaseResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def _load_checkpoint(path: Path) -> list[ExecutionCaseResult]:
    if not path.exists():
        return []

    results: list[ExecutionCaseResult] = []
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
            ExecutionCaseResult(
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
                estimated_raw_context_tokens=int(raw["estimated_raw_context_tokens"]),
                estimated_optimized_context_tokens=int(raw["estimated_optimized_context_tokens"]),
                estimated_context_reduction_ratio=float(raw["estimated_context_reduction_ratio"]),
                compressed_evidence_tokens=int(raw["compressed_evidence_tokens"]),
                tool_schema_tokens_planner=int(raw["tool_schema_tokens_planner"]),
                tool_schema_tokens_finalizer=int(raw["tool_schema_tokens_finalizer"]),
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
