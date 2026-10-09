from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Generic, TypeVar

from .context_models import ContextItem
from .policy_store import FilePolicyStore, PolicyNamespace
from .sdk import (
    ContextAdapter,
    OptimizationProfile,
    PortableContextOptimizer,
    PortablePayloadResult,
    PortableResult,
)
from .token_counter import TokenCounter


PayloadT = TypeVar("PayloadT")


class ProductContextRuntime(Generic[PayloadT]):
    """Product-scoped optimizer with persistent adaptive policy.

    One runtime owns exactly one product/profile namespace. This prevents
    learning from one product from silently changing another product's
    context selection policy.
    """

    def __init__(
        self,
        *,
        product: str,
        policy_store: FilePolicyStore,
        policy_profile: str = "default",
        token_counter: TokenCounter | None = None,
        default_profile: OptimizationProfile | None = None,
    ) -> None:
        self.namespace = PolicyNamespace(product, policy_profile)
        self.policy_store = policy_store
        self.token_counter = token_counter
        self.policy = policy_store.load(self.namespace)
        self.default_profile = default_profile or OptimizationProfile()
        self.optimizer = PortableContextOptimizer(
            token_counter=token_counter,
            learning_policy=self.policy,
        )

    def optimize(
        self,
        items: Sequence[ContextItem],
        *,
        profile: OptimizationProfile | None = None,
        known_fingerprints: Mapping[str, str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PortableResult:
        return self.optimizer.optimize(
            items,
            profile=profile or self.default_profile,
            known_fingerprints=known_fingerprints,
            metadata=self._metadata(metadata),
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
        return self.optimizer.optimize_payload(
            payload,
            adapter=adapter,
            profile=profile or self.default_profile,
            known_fingerprints=known_fingerprints,
            metadata=self._metadata(metadata),
        )

    def observe(
        self,
        *,
        items: Sequence[ContextItem],
        result: PortableResult,
        raw_quality_pass: bool,
        optimized_quality_pass: bool,
        attributed_keys: Iterable[str] | None = None,
        save: bool = True,
    ) -> int:
        touched = self.policy.observe(
            items=items,
            omitted_keys=result.omitted_keys,
            raw_quality_pass=raw_quality_pass,
            optimized_quality_pass=optimized_quality_pass,
            attributed_keys=attributed_keys,
        )
        if save and touched:
            self.save()
        return touched

    def observe_omission(
        self,
        *,
        items: Sequence[ContextItem],
        omitted_keys: Iterable[str],
        raw_quality_pass: bool,
        optimized_quality_pass: bool,
        attributed_keys: Iterable[str] | None = None,
        save: bool = True,
    ) -> int:
        touched = self.policy.observe(
            items=items,
            omitted_keys=omitted_keys,
            raw_quality_pass=raw_quality_pass,
            optimized_quality_pass=optimized_quality_pass,
            attributed_keys=attributed_keys,
        )
        if save and touched:
            self.save()
        return touched

    def save(self) -> None:
        self.policy_store.save(self.namespace, self.policy)

    def reset_policy(self) -> None:
        self.policy_store.delete(self.namespace)
        self.policy = self.policy.__class__()
        self.optimizer = PortableContextOptimizer(
            token_counter=self.token_counter,
            learning_policy=self.policy,
        )

    def policy_summary(self) -> dict[str, int]:
        return self.policy.summary()

    def _metadata(
        self,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        merged = {
            "product": self.namespace.product,
            "policy_profile": self.namespace.profile,
        }
        merged.update(dict(metadata or {}))
        return merged
