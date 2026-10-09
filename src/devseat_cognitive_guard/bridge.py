from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from .policy_store import FilePolicyStore
from .runtime import ProductContextRuntime
from .sdk import (
    SDK_SCHEMA_VERSION,
    OptimizationProfile,
    PortableContextOptimizer,
    item_from_mapping,
    portable_result_to_mapping,
)


def capabilities(*, persistent_policies: bool = False) -> dict[str, Any]:
    operations = ["capabilities", "optimize"]
    if persistent_policies:
        operations.extend(["feedback", "policy-summary"])
    return {
        "schema": SDK_SCHEMA_VERSION,
        "operations": operations,
        "persistent_policies": persistent_policies,
        "selection_strategies": ["priority", "density", "balanced"],
        "context_features": [
            "required",
            "stable",
            "source",
            "revision",
            "state_slot",
            "state_version",
            "dedup_blocks",
            "chunk_mode",
            "chunk_target_tokens",
        ],
        "chunk_modes": ["paragraph", "line"],
    }


def process_message(
    payload: Mapping[str, Any],
    *,
    optimizer: PortableContextOptimizer | None = None,
    policy_store: FilePolicyStore | None = None,
) -> dict[str, Any]:
    schema = str(payload.get("schema", SDK_SCHEMA_VERSION))
    if schema != SDK_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema {schema!r}; expected {SDK_SCHEMA_VERSION!r}"
        )

    operation = str(payload.get("op", "optimize"))
    if operation == "capabilities":
        return {
            "schema": SDK_SCHEMA_VERSION,
            "ok": True,
            "request_id": payload.get("request_id"),
            "capabilities": capabilities(
                persistent_policies=policy_store is not None
            ),
        }
    if operation == "optimize":
        return _process_optimize(
            payload,
            optimizer=optimizer,
            policy_store=policy_store,
        )
    if operation == "feedback":
        return _process_feedback(payload, policy_store=policy_store)
    if operation == "policy-summary":
        return _process_policy_summary(payload, policy_store=policy_store)
    raise ValueError(f"unsupported operation: {operation!r}")


def _process_optimize(
    payload: Mapping[str, Any],
    *,
    optimizer: PortableContextOptimizer | None,
    policy_store: FilePolicyStore | None,
) -> dict[str, Any]:
    items = _parse_items(payload)
    profile = _parse_profile(payload)
    known = payload.get("known_context_fingerprints") or {}
    if not isinstance(known, Mapping):
        raise ValueError("known_context_fingerprints must be an object")

    product = _optional_product(payload)
    policy_profile = str(payload.get("policy_profile", "default"))
    metadata = {
        "request_id": payload.get("request_id"),
        "product": product,
        "policy_profile": policy_profile,
    }

    runtime = None
    if policy_store is not None and product is not None:
        runtime = ProductContextRuntime(
            product=product,
            policy_profile=policy_profile,
            policy_store=policy_store,
            default_profile=profile,
        )
        result = runtime.optimize(
            items,
            known_fingerprints={str(k): str(v) for k, v in known.items()},
            metadata=metadata,
        )
    else:
        runner = optimizer or PortableContextOptimizer()
        result = runner.optimize(
            items,
            profile=profile,
            known_fingerprints={str(k): str(v) for k, v in known.items()},
            metadata=metadata,
        )

    response = portable_result_to_mapping(result)
    response["request_id"] = payload.get("request_id")
    response["ok"] = True
    response["policy_mode"] = (
        "persistent" if runtime is not None else "off"
    )
    if runtime is not None:
        response["policy_summary"] = runtime.policy_summary()
    return response


def _process_feedback(
    payload: Mapping[str, Any],
    *,
    policy_store: FilePolicyStore | None,
) -> dict[str, Any]:
    if policy_store is None:
        raise ValueError("feedback requires bridge --policy-root")
    product = _required_product(payload)
    policy_profile = str(payload.get("policy_profile", "default"))
    items = _parse_items(payload)

    omitted = payload.get("omitted_keys")
    if not isinstance(omitted, list):
        raise ValueError("omitted_keys must be a JSON array")
    attributed = payload.get("attributed_keys")
    if attributed is not None and not isinstance(attributed, list):
        raise ValueError("attributed_keys must be a JSON array or null")
    if "raw_quality_pass" not in payload:
        raise ValueError("raw_quality_pass is required")
    if "optimized_quality_pass" not in payload:
        raise ValueError("optimized_quality_pass is required")

    runtime = ProductContextRuntime(
        product=product,
        policy_profile=policy_profile,
        policy_store=policy_store,
    )
    touched = runtime.observe_omission(
        items=items,
        omitted_keys=(str(key) for key in omitted),
        raw_quality_pass=bool(payload["raw_quality_pass"]),
        optimized_quality_pass=bool(payload["optimized_quality_pass"]),
        attributed_keys=(
            None
            if attributed is None
            else (str(key) for key in attributed)
        ),
    )
    return {
        "schema": SDK_SCHEMA_VERSION,
        "ok": True,
        "request_id": payload.get("request_id"),
        "product": product,
        "policy_profile": policy_profile,
        "features_touched": touched,
        "policy_summary": runtime.policy_summary(),
    }


def _process_policy_summary(
    payload: Mapping[str, Any],
    *,
    policy_store: FilePolicyStore | None,
) -> dict[str, Any]:
    if policy_store is None:
        raise ValueError("policy-summary requires bridge --policy-root")
    product = _required_product(payload)
    policy_profile = str(payload.get("policy_profile", "default"))
    runtime = ProductContextRuntime(
        product=product,
        policy_profile=policy_profile,
        policy_store=policy_store,
    )
    return {
        "schema": SDK_SCHEMA_VERSION,
        "ok": True,
        "request_id": payload.get("request_id"),
        "product": product,
        "policy_profile": policy_profile,
        "policy_summary": runtime.policy_summary(),
    }


def _parse_items(payload: Mapping[str, Any]):
    raw_context = payload.get("context")
    if not isinstance(raw_context, list):
        raise ValueError("context must be a JSON array")
    return tuple(item_from_mapping(item) for item in raw_context)


def _parse_profile(payload: Mapping[str, Any]) -> OptimizationProfile:
    raw_profile = payload.get("profile") or {}
    if not isinstance(raw_profile, Mapping):
        raise ValueError("profile must be a JSON object")
    return OptimizationProfile(
        token_budget=(
            None
            if raw_profile.get("token_budget") is None
            else int(raw_profile["token_budget"])
        ),
        char_budget=(
            None
            if raw_profile.get("char_budget") is None
            else int(raw_profile["char_budget"])
        ),
        selection_strategy=str(
            raw_profile.get("selection_strategy", "balanced")
        ),
        delta_context=bool(raw_profile.get("delta_context", False)),
    )


def _optional_product(payload: Mapping[str, Any]) -> str | None:
    product = payload.get("product")
    if product is None:
        return None
    value = str(product).strip()
    return value or None


def _required_product(payload: Mapping[str, Any]) -> str:
    product = _optional_product(payload)
    if product is None:
        raise ValueError("product is required for persistent policy operations")
    return product


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="context-bridge",
        description=(
            "Language-neutral JSONL bridge for DevSeat Context Optimizer."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit on the first invalid input line.",
    )
    parser.add_argument(
        "--policy-root",
        default=os.getenv("DEVSEAT_CONTEXT_POLICY_ROOT"),
        help=(
            "Enable product-scoped persistent learning policies under this "
            "directory. Can also be set with DEVSEAT_CONTEXT_POLICY_ROOT."
        ),
    )
    args = parser.parse_args()

    optimizer = PortableContextOptimizer()
    policy_store = (
        FilePolicyStore(Path(args.policy_root))
        if args.policy_root
        else None
    )

    for line_number, line in enumerate(sys.stdin, start=1):
        if not line.strip():
            continue
        request_id = None
        try:
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError("each JSONL line must be an object")
            request_id = payload.get("request_id")
            response = process_message(
                payload,
                optimizer=optimizer,
                policy_store=policy_store,
            )
        except Exception as exc:
            if args.fail_fast:
                raise
            response = {
                "schema": SDK_SCHEMA_VERSION,
                "ok": False,
                "request_id": request_id,
                "error": str(exc),
                "line": line_number,
            }

        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
