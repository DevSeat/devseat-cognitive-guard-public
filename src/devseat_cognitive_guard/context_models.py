from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ContextItem:
    key: str
    content: str
    priority: int = 50
    required: bool = False
    stable: bool = False
    source: str | None = None
    revision: str | None = None
    state_slot: str | None = None
    state_version: int | None = None
    dedup_blocks: bool = False
    chunk_mode: str | None = None
    chunk_target_tokens: int | None = None


@dataclass(frozen=True)
class ReducedContext:
    items: tuple[ContextItem, ...]
    omitted_keys: tuple[str, ...]
    duplicate_keys: tuple[str, ...]
    original_chars: int
    reduced_chars: int
    original_tokens: int = 0
    reduced_tokens: int = 0
    unchanged_keys: tuple[str, ...] = ()
    superseded_keys: tuple[str, ...] = ()
    compacted_keys: tuple[str, ...] = ()
    chunked_keys: tuple[str, ...] = ()
    stage_token_savings: Mapping[str, int] = field(default_factory=dict)
    context_fingerprints: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextRequest:
    context: Sequence[ContextItem] = field(default_factory=tuple)
    context_budget_chars: int | None = 12000
    context_budget_tokens: int | None = None
    delta_context: bool = False
    known_context_fingerprints: Mapping[str, str] = field(default_factory=dict)
    selection_strategy: str = "priority"


@dataclass(frozen=True)
class OptimizedContext:
    reduced_context: ReducedContext
    context_fingerprints: Mapping[str, str]
    original_context_chars: int
    reduced_context_chars: int
    reduction_ratio: float
    original_context_tokens: int = 0
    reduced_context_tokens: int = 0
    saved_context_tokens: int = 0
    token_reduction_ratio: float = 0.0
    stage_token_savings: Mapping[str, int] = field(default_factory=dict)
    notes: tuple[str, ...] = ()
