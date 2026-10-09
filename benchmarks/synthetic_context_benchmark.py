from __future__ import annotations

import argparse
import json
import platform
from time import perf_counter

from devseat_cognitive_guard import ContextItem, ContextOptimizer, ContextRequest


def run(cases: int, noise_blocks: int, budget: int) -> dict[str, object]:
    optimizer = ContextOptimizer()
    original_tokens = 0
    reduced_tokens = 0
    preserved = 0
    started = perf_counter()

    for case_number in range(cases):
        required_key = f"required-{case_number}"
        items = [
            ContextItem(
                key=required_key,
                content=f"Synthetic case {case_number}: preserve marker {case_number}.",
                priority=100,
                required=True,
            )
        ]
        for block_number in range(noise_blocks):
            content = (
                f"Synthetic repeated background block {block_number % 4}. "
                "This low-priority context is generated for a local benchmark."
            )
            items.append(
                ContextItem(
                    key=f"noise-{case_number}-{block_number}",
                    content=content,
                    priority=block_number % 10,
                )
            )

        result = optimizer.optimize(
            ContextRequest(context=items, context_budget_tokens=budget)
        )
        original_tokens += result.original_context_tokens
        reduced_tokens += result.reduced_context_tokens
        preserved += any(
            item.key == required_key for item in result.reduced_context.items
        )

    elapsed = perf_counter() - started
    saved_tokens = original_tokens - reduced_tokens
    ratio = saved_tokens / original_tokens if original_tokens else 0.0
    return {
        "workload": "synthetic_local_context",
        "network_requests": 0,
        "model_calls": 0,
        "python": platform.python_version(),
        "cases": cases,
        "noise_blocks_per_case": noise_blocks,
        "token_budget": budget,
        "required_items_preserved": preserved,
        "original_estimated_tokens": original_tokens,
        "reduced_estimated_tokens": reduced_tokens,
        "saved_estimated_tokens": saved_tokens,
        "estimated_reduction_ratio": round(ratio, 6),
        "elapsed_seconds": round(elapsed, 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a deterministic, network-free synthetic context benchmark."
    )
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--noise-blocks", type=int, default=20)
    parser.add_argument("--budget", type=int, default=128)
    args = parser.parse_args()
    if args.cases < 1 or args.noise_blocks < 1 or args.budget < 1:
        parser.error("cases, noise-blocks, and budget must be positive")
    print(
        json.dumps(
            run(args.cases, args.noise_blocks, args.budget),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
