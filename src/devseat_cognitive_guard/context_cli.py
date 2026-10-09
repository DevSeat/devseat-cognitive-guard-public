from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .context_models import ContextItem, ContextRequest
from .context_optimizer import ContextOptimizer


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="context-reduce",
        description="Reduce model context without rewriting prompts or calling a model.",
    )
    parser.add_argument("input", nargs="?", default="-", help="JSON file or '-' for stdin.")
    parser.add_argument("--output", default="-", help="JSON file or '-' for stdout.")
    args = parser.parse_args()

    optimized = ContextOptimizer().optimize(_parse_request(_read_json(args.input)))
    rendered = json.dumps(_serialize(optimized), ensure_ascii=False, indent=2)

    if args.output == "-":
        print(rendered)
    else:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0


def _read_json(path: str) -> dict[str, Any]:
    if path == "-":
        return json.load(sys.stdin)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_request(payload: dict[str, Any]) -> ContextRequest:
    raw_items = payload.get("context", ())
    if not isinstance(raw_items, list):
        raise ValueError("context must be a JSON array")

    items = tuple(
        ContextItem(
            key=str(item["key"]),
            content=str(item["content"]),
            priority=int(item.get("priority", 50)),
            required=bool(item.get("required", False)),
            stable=bool(item.get("stable", False)),
            source=None if item.get("source") is None else str(item["source"]),
            revision=None if item.get("revision") is None else str(item["revision"]),
            state_slot=(
                None if item.get("state_slot") is None else str(item["state_slot"])
            ),
            state_version=(
                None if item.get("state_version") is None else int(item["state_version"])
            ),
            dedup_blocks=bool(item.get("dedup_blocks", False)),
            chunk_mode=(
                None if item.get("chunk_mode") is None else str(item["chunk_mode"])
            ),
            chunk_target_tokens=(
                None
                if item.get("chunk_target_tokens") is None
                else int(item["chunk_target_tokens"])
            ),
        )
        for item in raw_items
    )

    known = payload.get("known_context_fingerprints", {})
    if not isinstance(known, dict):
        raise ValueError("known_context_fingerprints must be a JSON object")

    raw_char_budget = payload.get("context_budget_chars", 12000)
    raw_token_budget = payload.get("context_budget_tokens")

    return ContextRequest(
        context=items,
        context_budget_chars=(
            None if raw_char_budget is None else int(raw_char_budget)
        ),
        context_budget_tokens=(
            None if raw_token_budget is None else int(raw_token_budget)
        ),
        delta_context=bool(payload.get("delta_context", False)),
        known_context_fingerprints={str(k): str(v) for k, v in known.items()},
        selection_strategy=str(payload.get("selection_strategy", "priority")),
    )


def _serialize(optimized) -> dict[str, Any]:
    reduced = optimized.reduced_context
    return {
        "context": [
            {
                "key": item.key,
                "content": item.content,
                "priority": item.priority,
                "required": item.required,
                "stable": item.stable,
                "source": item.source,
                "revision": item.revision,
                "state_slot": item.state_slot,
                "state_version": item.state_version,
                "dedup_blocks": item.dedup_blocks,
                "chunk_mode": item.chunk_mode,
                "chunk_target_tokens": item.chunk_target_tokens,
            }
            for item in reduced.items
        ],
        "omitted_keys": list(reduced.omitted_keys),
        "duplicate_keys": list(reduced.duplicate_keys),
        "unchanged_keys": list(reduced.unchanged_keys),
        "superseded_keys": list(reduced.superseded_keys),
        "compacted_keys": list(reduced.compacted_keys),
        "chunked_keys": list(reduced.chunked_keys),
        "context_fingerprints": dict(optimized.context_fingerprints),
        "original_context_chars": optimized.original_context_chars,
        "reduced_context_chars": optimized.reduced_context_chars,
        "reduction_ratio": optimized.reduction_ratio,
        "original_context_tokens": optimized.original_context_tokens,
        "reduced_context_tokens": optimized.reduced_context_tokens,
        "saved_context_tokens": optimized.saved_context_tokens,
        "token_reduction_ratio": optimized.token_reduction_ratio,
        "stage_token_savings": dict(optimized.stage_token_savings),
        "notes": list(optimized.notes),
    }


if __name__ == "__main__":
    raise SystemExit(main())
