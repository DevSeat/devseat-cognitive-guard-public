from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .context_models import ContextItem


POLICY_VERSION = 1


@dataclass
class FeatureStats:
    bias: int = 0
    safe_omissions: int = 0
    regressions: int = 0


@dataclass
class AdaptiveContextPolicy:
    """Persistent bounded online policy for optional-context selection.

    The policy never changes required semantics and never stores context text.
    It only learns priority adjustments from metadata-derived feature buckets.
    """

    features: dict[str, FeatureStats] = field(default_factory=dict)
    regression_step: int = 8
    success_step: int = 1
    min_bias: int = -24
    max_bias: int = 48

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "AdaptiveContextPolicy":
        version = int(payload.get("version", 0))
        if version != POLICY_VERSION:
            raise ValueError(
                f"unsupported adaptive policy version: {version}"
            )

        policy = cls(
            regression_step=int(payload.get("regression_step", 8)),
            success_step=int(payload.get("success_step", 1)),
            min_bias=int(payload.get("min_bias", -24)),
            max_bias=int(payload.get("max_bias", 48)),
        )
        raw_features = payload.get("features", {})
        if not isinstance(raw_features, Mapping):
            raise ValueError("policy features must be an object")

        for name, raw in raw_features.items():
            if not isinstance(raw, Mapping):
                continue
            policy.features[str(name)] = FeatureStats(
                bias=int(raw.get("bias", 0)),
                safe_omissions=int(raw.get("safe_omissions", 0)),
                regressions=int(raw.get("regressions", 0)),
            )
        return policy

    @classmethod
    def load(cls, path: str | Path) -> "AdaptiveContextPolicy":
        target = Path(path)
        if not target.exists():
            return cls()

        payload = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("policy document must be an object")
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(target)

    def summary(self) -> dict[str, int]:
        biases = [stats.bias for stats in self.features.values()]
        return {
            "feature_count": len(self.features),
            "positive_bias_features": sum(value > 0 for value in biases),
            "negative_bias_features": sum(value < 0 for value in biases),
            "zero_bias_features": sum(value == 0 for value in biases),
            "total_safe_omissions": sum(
                stats.safe_omissions for stats in self.features.values()
            ),
            "total_regressions": sum(
                stats.regressions for stats in self.features.values()
            ),
            "max_bias": max(biases, default=0),
            "min_bias": min(biases, default=0),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": POLICY_VERSION,
            "regression_step": self.regression_step,
            "success_step": self.success_step,
            "min_bias": self.min_bias,
            "max_bias": self.max_bias,
            "features": {
                name: {
                    "bias": stats.bias,
                    "safe_omissions": stats.safe_omissions,
                    "regressions": stats.regressions,
                }
                for name, stats in sorted(self.features.items())
            },
        }

    def apply(
        self,
        items: Sequence[ContextItem],
    ) -> tuple[ContextItem, ...]:
        adjusted: list[ContextItem] = []
        for item in items:
            if item.required:
                adjusted.append(item)
                continue

            bias = self.priority_bias(item)
            if bias == 0:
                adjusted.append(item)
                continue

            adjusted.append(
                replace(
                    item,
                    priority=max(0, min(200, item.priority + bias)),
                )
            )
        return tuple(adjusted)

    def priority_bias(self, item: ContextItem) -> int:
        scores = [
            self.features[name].bias
            for name in feature_names(item)
            if name in self.features
        ]
        if not scores:
            return 0

        # Additive evidence is useful across source + state slot + key family,
        # but the final adjustment is bounded to avoid runaway drift.
        return max(self.min_bias, min(self.max_bias, sum(scores)))

    def observe(
        self,
        *,
        items: Sequence[ContextItem],
        omitted_keys: Iterable[str],
        raw_quality_pass: bool,
        optimized_quality_pass: bool,
        attributed_keys: Iterable[str] | None = None,
    ) -> int:
        """Learn only from comparable RAW-success outcomes.

        RAW failure is not evidence about compression, so it produces no
        update. Optimized regressions strongly increase keep-bias; successful
        omissions slowly decrease it.
        """

        if not raw_quality_pass:
            return 0

        omitted = set(omitted_keys)
        attributed = (
            None
            if attributed_keys is None
            else set(attributed_keys)
        )
        candidate_keys = (
            omitted
            if attributed is None
            else omitted.intersection(attributed)
        )
        candidates = [
            item
            for item in items
            if item.key in candidate_keys and not item.required
        ]
        if not candidates:
            return 0

        if optimized_quality_pass:
            delta = -abs(self.success_step)
            regression = False
        else:
            delta = abs(self.regression_step)
            regression = True

        touched = 0
        seen_features: set[str] = set()
        for item in candidates:
            for feature in feature_names(item):
                if feature in seen_features:
                    continue
                seen_features.add(feature)
                stats = self.features.setdefault(feature, FeatureStats())
                stats.bias = max(
                    self.min_bias,
                    min(self.max_bias, stats.bias + delta),
                )
                if regression:
                    stats.regressions += 1
                else:
                    stats.safe_omissions += 1
                touched += 1
        return touched


def feature_names(item: ContextItem) -> tuple[str, ...]:
    features: list[str] = []

    if item.source:
        features.append(f"source:{item.source}")

    if item.state_slot:
        features.append(f"slot:{item.state_slot}")

    family = _key_family(item.key)
    if family:
        features.append(f"key-family:{family}")

    # Preserve order but avoid duplicate feature names.
    return tuple(dict.fromkeys(features))


def _key_family(key: str) -> str:
    normalized = re.sub(r"\d+", "#", key.strip().casefold())
    normalized = re.sub(r"[^a-z0-9가-힣_.:#-]+", "-", normalized)
    return normalized[:80]
