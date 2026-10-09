from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Protocol, Sequence, TypeVar

from .context_models import ContextItem, ContextRequest, OptimizedContext
from .context_optimizer import ContextOptimizer
from .learning_policy import AdaptiveContextPolicy
from .token_counter import TokenCounter


SDK_SCHEMA_VERSION = "devseat.context.v1"
PayloadT = TypeVar("PayloadT")


class ContextAdapter(Protocol[PayloadT]):
    """Convert one product payload to/from the portable ContextItem contract."""

    def to_items(self, payload: PayloadT) -> Sequence[ContextItem]:
        ...

    def from_items(
        self,
        original: PayloadT,
        selected: Sequence[ContextItem],
    ) -> PayloadT:
        ...


@dataclass(frozen=True)
class PortablePayloadResult(Generic[PayloadT]):
    payload: PayloadT
    optimization: "PortableResult"

@dataclass(frozen=True)
class OptimizationProfile:
    """Stable product-facing optimization policy."""

    token_budget: int | None = None
    char_budget: int | None = None
    selection_strategy: str = "balanced"
    delta_context: bool = False


@dataclass(frozen=True)
class PortableResult:
    optimized: OptimizedContext
    schema: str = SDK_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def items(self) -> tuple[ContextItem, ...]:
        return self.optimized.reduced_context.items

    @property
    def omitted_keys(self) -> tuple[str, ...]:
        return self.optimized.reduced_context.omitted_keys


class PortableContextOptimizer:
    """Stable embedding API for DevSeat products and external programs."""

    def __init__(
        self,
        *,
        token_counter: TokenCounter | None = None,
        learning_policy: AdaptiveContextPolicy | None = None,
    ) -> None:
        self._optimizer = ContextOptimizer(
            token_counter=token_counter,
            learning_policy=learning_policy,
        )

    def optimize_payload(
        self,
        payload: PayloadT,
        *,
        adapter: ContextAdapter[PayloadT],
        profile: OptimizationProfile | None = None,
        known_fingerprints: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PortablePayloadResult[PayloadT]:
        result = self.optimize(
            adapter.to_items(payload),
            profile=profile,
            known_fingerprints=known_fingerprints,
            metadata=metadata,
        )
        return PortablePayloadResult(
            payload=adapter.from_items(payload, result.items),
            optimization=result,
        )

    def optimize(
        self,
        items: Sequence[ContextItem],
        *,
        profile: OptimizationProfile | None = None,
        known_fingerprints: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PortableResult:
        selected = profile or OptimizationProfile()
        optimized = self._optimizer.optimize(
            ContextRequest(
                context=tuple(items),
                context_budget_chars=selected.char_budget,
                context_budget_tokens=selected.token_budget,
                delta_context=selected.delta_context,
                known_context_fingerprints=known_fingerprints or {},
                selection_strategy=selected.selection_strategy,
            )
        )
        return PortableResult(
            optimized=optimized,
            metadata=dict(metadata or {}),
        )


def item_from_mapping(raw: Mapping[str, Any]) -> ContextItem:
    return ContextItem(
        key=str(raw["key"]),
        content=str(raw.get("content", "")),
        priority=int(raw.get("priority", 50)),
        required=bool(raw.get("required", False)),
        stable=bool(raw.get("stable", False)),
        source=None if raw.get("source") is None else str(raw["source"]),
        revision=None if raw.get("revision") is None else str(raw["revision"]),
        state_slot=(
            None if raw.get("state_slot") is None else str(raw["state_slot"])
        ),
        state_version=(
            None
            if raw.get("state_version") is None
            else int(raw["state_version"])
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


def item_to_mapping(item: ContextItem) -> dict[str, Any]:
    return {
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


def portable_result_to_mapping(result: PortableResult) -> dict[str, Any]:
    optimized = result.optimized
    reduced = optimized.reduced_context
    return {
        "schema": result.schema,
        "metadata": dict(result.metadata),
        "context": [item_to_mapping(item) for item in reduced.items],
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
