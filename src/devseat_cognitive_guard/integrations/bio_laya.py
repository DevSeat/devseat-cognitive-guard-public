from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..context_models import ContextItem
from ..sdk import OptimizationProfile, PortableContextOptimizer, PortableResult
from ..token_adapters import HuggingFaceTokenizerCounter


@dataclass(frozen=True)
class BioStateResult:
    state: str | dict | list
    optimization: PortableResult


class BioLayaContextOptimizer:
    """Reference adapter for BIO/Laya `state` values.

    The adapter preserves the host state's Python type. ContextItem content is
    only an optimization view; selected values are reconstructed from the
    original state so structured BIO payloads do not suffer serialization
    round-trips or type drift.
    """

    def __init__(
        self,
        *,
        optimizer: PortableContextOptimizer | None = None,
        required_fields: Sequence[str] = (),
        stable_fields: Sequence[str] = (),
        field_priorities: Mapping[str, int] | None = None,
        keep_last_turns: int = 4,
    ) -> None:
        self.optimizer = optimizer or PortableContextOptimizer()
        self.required_fields = set(required_fields)
        self.stable_fields = set(stable_fields)
        self.field_priorities = {
            str(key): int(value)
            for key, value in (field_priorities or {}).items()
        }
        self.keep_last_turns = max(0, int(keep_last_turns))

    @classmethod
    def from_laya_agent(
        cls,
        agent: Any,
        **kwargs,
    ) -> "BioLayaContextOptimizer":
        """Use the already-loaded Laya tokenizer for exact host accounting."""

        optimizer = PortableContextOptimizer(
            token_counter=HuggingFaceTokenizerCounter(agent.tok)
        )
        return cls(optimizer=optimizer, **kwargs)

    def optimize_state(
        self,
        state: str | dict | list,
        *,
        token_budget: int | None,
        selection_strategy: str = "balanced",
    ) -> BioStateResult:
        if isinstance(state, str):
            items = (
                ContextItem(
                    key="bio:string",
                    content=state,
                    priority=100,
                    required=True,
                    source="bio-string",
                ),
            )
            result = self.optimizer.optimize(
                items,
                profile=OptimizationProfile(
                    token_budget=token_budget,
                    selection_strategy=selection_strategy,
                ),
                metadata={"adapter": "bio-laya", "state_type": "str"},
            )
            return BioStateResult(state=state, optimization=result)

        if isinstance(state, dict):
            return self._optimize_dict(
                state,
                token_budget=token_budget,
                selection_strategy=selection_strategy,
            )

        if isinstance(state, list):
            return self._optimize_list(
                state,
                token_budget=token_budget,
                selection_strategy=selection_strategy,
            )

        raise TypeError("BIO/Laya state must be str, dict, or list")

    def _optimize_dict(
        self,
        state: dict,
        *,
        token_budget: int | None,
        selection_strategy: str,
    ) -> BioStateResult:
        items: list[ContextItem] = []
        key_map: dict[str, Any] = {}

        for index, (field, value) in enumerate(state.items()):
            field_name = str(field)
            item_key = f"bio:field:{index:06d}"
            key_map[item_key] = field
            items.append(
                ContextItem(
                    key=item_key,
                    content=_serialize_value(value),
                    priority=self.field_priorities.get(field_name, 50),
                    required=field_name in self.required_fields,
                    stable=field_name in self.stable_fields,
                    source=f"bio-field:{field_name}",
                )
            )

        result = self.optimizer.optimize(
            items,
            profile=OptimizationProfile(
                token_budget=token_budget,
                selection_strategy=selection_strategy,
            ),
            metadata={"adapter": "bio-laya", "state_type": "dict"},
        )
        selected_keys = {item.key for item in result.items}
        compacted = {
            original_key: state[original_key]
            for item_key, original_key in key_map.items()
            if item_key in selected_keys
        }
        return BioStateResult(state=compacted, optimization=result)

    def _optimize_list(
        self,
        state: list,
        *,
        token_budget: int | None,
        selection_strategy: str,
    ) -> BioStateResult:
        items: list[ContextItem] = []
        index_map: dict[str, int] = {}
        required_from = max(0, len(state) - self.keep_last_turns)

        for index, value in enumerate(state):
            item_key = f"bio:turn:{index:06d}"
            index_map[item_key] = index
            is_recent = index >= required_from
            items.append(
                ContextItem(
                    key=item_key,
                    content=_serialize_value(value),
                    priority=100 if is_recent else 50,
                    required=is_recent,
                    stable=False,
                    source="bio-turn",
                )
            )

        result = self.optimizer.optimize(
            items,
            profile=OptimizationProfile(
                token_budget=token_budget,
                selection_strategy=selection_strategy,
            ),
            metadata={"adapter": "bio-laya", "state_type": "list"},
        )
        selected = {item.key for item in result.items}
        compacted = [
            state[index]
            for item_key, index in index_map.items()
            if item_key in selected
        ]
        return BioStateResult(state=compacted, optimization=result)


def _serialize_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


class OptimizingLayaRouter:
    """Drop-in wrapper around an existing Laya Router instance.

    This depends only on the Router public methods route/load and the loaded
    agent system_one method. The integration module does not import Laya, so
    devseat-cognitive-guard remains dependency-free.
    """

    def __init__(
        self,
        router: Any,
        *,
        token_budget: int,
        required_fields: Sequence[str] = (),
        stable_fields: Sequence[str] = (),
        field_priorities: Mapping[str, int] | None = None,
        keep_last_turns: int = 4,
        selection_strategy: str = "balanced",
    ) -> None:
        self.router = router
        self.token_budget = int(token_budget)
        self.required_fields = tuple(required_fields)
        self.stable_fields = tuple(stable_fields)
        self.field_priorities = dict(field_priorities or {})
        self.keep_last_turns = int(keep_last_turns)
        self.selection_strategy = selection_strategy

    def predict(
        self,
        state: str | dict | list,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        task: str | None = None,
        lang: str | None = None,
    ) -> dict[str, Any]:
        decision = self.router.route(
            state,
            dict(questions),
            model=model,
            task=task,
            lang=lang,
        )
        agent = self.router.load(decision["model"])
        adapter = BioLayaContextOptimizer.from_laya_agent(
            agent,
            required_fields=self.required_fields,
            stable_fields=self.stable_fields,
            field_priorities=self.field_priorities,
            keep_last_turns=self.keep_last_turns,
        )
        compacted = adapter.optimize_state(
            state,
            token_budget=self.token_budget,
            selection_strategy=self.selection_strategy,
        )
        result = agent.system_one(compacted.state, dict(questions))
        result["routing"] = dict(decision)
        optimized = compacted.optimization.optimized
        result["context_optimization"] = {
            "schema": compacted.optimization.schema,
            "original_context_tokens": optimized.original_context_tokens,
            "reduced_context_tokens": optimized.reduced_context_tokens,
            "saved_context_tokens": optimized.saved_context_tokens,
            "token_reduction_ratio": optimized.token_reduction_ratio,
            "omitted_keys": list(
                optimized.reduced_context.omitted_keys
            ),
        }
        return result

    system_one = predict
