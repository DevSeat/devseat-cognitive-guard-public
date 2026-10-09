from __future__ import annotations

from .context_models import ContextRequest, OptimizedContext
from .context_reducer import ContextReducer
from .learning_policy import AdaptiveContextPolicy
from .token_counter import ApproxTokenCounter, TokenCounter


class ContextOptimizer:
    """Model-agnostic deterministic context reduction."""

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        learning_policy: AdaptiveContextPolicy | None = None,
    ) -> None:
        self.reducer = ContextReducer()
        self.token_counter = token_counter or ApproxTokenCounter()
        self.learning_policy = learning_policy

    def optimize(self, request: ContextRequest) -> OptimizedContext:
        context = (
            self.learning_policy.apply(request.context)
            if self.learning_policy is not None
            else request.context
        )
        reduced = self.reducer.reduce(
            context,
            budget_chars=request.context_budget_chars,
            budget_tokens=request.context_budget_tokens,
            token_counter=self.token_counter,
            delta_mode=request.delta_context,
            known_fingerprints=request.known_context_fingerprints,
            selection_strategy=request.selection_strategy,
        )

        char_ratio = 0.0
        if reduced.original_chars:
            char_ratio = 1.0 - (reduced.reduced_chars / reduced.original_chars)

        token_ratio = 0.0
        if reduced.original_tokens:
            token_ratio = 1.0 - (reduced.reduced_tokens / reduced.original_tokens)

        saved_tokens = reduced.original_tokens - reduced.reduced_tokens

        notes: list[str] = []
        if reduced.duplicate_keys:
            notes.append(
                f"removed {len(reduced.duplicate_keys)} exact duplicate context items"
            )
        if reduced.omitted_keys:
            notes.append(
                f"omitted {len(reduced.omitted_keys)} lower-priority context items"
            )
        if reduced.unchanged_keys:
            notes.append(
                f"omitted {len(reduced.unchanged_keys)} unchanged stable context items"
            )
        if reduced.superseded_keys:
            notes.append(
                f"removed {len(reduced.superseded_keys)} superseded state items"
            )
        if reduced.compacted_keys:
            notes.append(
                f"compacted repeated blocks in {len(reduced.compacted_keys)} items"
            )
        if reduced.chunked_keys:
            notes.append(
                f"split {len(reduced.chunked_keys)} oversized optional items into token-aware chunks"
            )
        if request.selection_strategy == "density":
            notes.append("selected optional context by priority-per-token density")
        if request.selection_strategy == "balanced":
            notes.append(
                "selected optional context by priority bands with token-density packing"
            )
        if self.learning_policy is not None:
            notes.append("applied adaptive learned priority policy")
        if saved_tokens > 0:
            notes.append(
                f"saved approximately {saved_tokens} context tokens"
            )

        return OptimizedContext(
            reduced_context=reduced,
            context_fingerprints=reduced.context_fingerprints,
            original_context_chars=reduced.original_chars,
            reduced_context_chars=reduced.reduced_chars,
            reduction_ratio=char_ratio,
            original_context_tokens=reduced.original_tokens,
            reduced_context_tokens=reduced.reduced_tokens,
            saved_context_tokens=saved_tokens,
            token_reduction_ratio=token_ratio,
            stage_token_savings=reduced.stage_token_savings,
            notes=tuple(notes),
        )
