from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import replace

from .context_models import ContextItem, ReducedContext
from .token_counter import ApproxTokenCounter, TokenCounter


SELECTION_STRATEGIES = {"priority", "density", "balanced"}
CHUNK_MODES = {"paragraph", "line"}


class ContextReducer:
    """Deterministically reduce context without semantic rewriting."""

    def reduce(
        self,
        items: Sequence[ContextItem],
        *,
        budget_chars: int | None = 12000,
        budget_tokens: int | None = None,
        token_counter: TokenCounter | None = None,
        delta_mode: bool = False,
        known_fingerprints: Mapping[str, str] | None = None,
        selection_strategy: str = "priority",
    ) -> ReducedContext:
        if selection_strategy not in SELECTION_STRATEGIES:
            raise ValueError(
                f"selection_strategy must be one of {sorted(SELECTION_STRATEGIES)}"
            )

        _validate_items(items)

        counter = token_counter or ApproxTokenCounter()
        char_budget = None if budget_chars is None else max(1, int(budget_chars))
        token_budget = None if budget_tokens is None else max(1, int(budget_tokens))
        original_chars = sum(len(item.content) for item in items)
        original_tokens = sum(counter.count(item.content) for item in items)
        known = known_fingerprints or {}

        # Fingerprints always describe original caller-visible items.
        fingerprints = {
            item.key: context_fingerprint(item)
            for item in items
        }

        compacted, compacted_keys = _compact_opt_in_blocks(items)
        compacted_tokens = sum(counter.count(item.content) for item in compacted)

        unique, duplicate_keys = _deduplicate(compacted)
        unique_tokens = sum(counter.count(item.content) for item in unique)

        current, superseded_keys = _remove_superseded_state(unique)
        current_tokens = sum(counter.count(item.content) for item in current)

        retained: list[ContextItem] = []
        unchanged_keys: list[str] = []
        for item in current:
            if (
                delta_mode
                and item.stable
                and known.get(item.key) == fingerprints[item.key]
            ):
                unchanged_keys.append(item.key)
                continue
            retained.append(item)

        retained_tokens = sum(counter.count(item.content) for item in retained)

        chunked, chunked_keys = _chunk_optional_items(
            retained,
            counter=counter,
            reserved_keys={item.key for item in items},
        )

        required = [item for item in chunked if item.required]
        optional = [item for item in chunked if not item.required]
        optional.sort(
            key=lambda item: _selection_key(
                item,
                strategy=selection_strategy,
                counter=counter,
            )
        )

        selected: list[ContextItem] = list(required)
        used_chars = sum(len(item.content) for item in required)
        used_tokens = sum(counter.count(item.content) for item in required)
        omitted: list[str] = []

        for item in optional:
            item_chars = len(item.content)
            item_tokens = counter.count(item.content)

            if token_budget is not None:
                fits = used_tokens + item_tokens <= token_budget
            elif char_budget is not None:
                fits = used_chars + item_chars <= char_budget
            else:
                fits = True

            if fits:
                selected.append(item)
                used_chars += item_chars
                used_tokens += item_tokens
            else:
                omitted.append(item.key)

        reduced_tokens = sum(counter.count(item.content) for item in selected)
        stage_token_savings = {
            "block_dedup": max(0, original_tokens - compacted_tokens),
            "item_dedup": max(0, compacted_tokens - unique_tokens),
            "supersession": max(0, unique_tokens - current_tokens),
            "delta": max(0, current_tokens - retained_tokens),
            "budget": max(0, retained_tokens - reduced_tokens),
        }

        return ReducedContext(
            items=tuple(selected),
            omitted_keys=tuple(omitted),
            duplicate_keys=tuple(duplicate_keys),
            original_chars=original_chars,
            reduced_chars=sum(len(item.content) for item in selected),
            original_tokens=original_tokens,
            reduced_tokens=reduced_tokens,
            unchanged_keys=tuple(unchanged_keys),
            superseded_keys=tuple(superseded_keys),
            compacted_keys=tuple(compacted_keys),
            chunked_keys=tuple(chunked_keys),
            stage_token_savings=stage_token_savings,
            context_fingerprints=fingerprints,
        )


def context_fingerprint(item: ContextItem) -> str:
    # Preserve all meaningful internal whitespace. Only newline encoding is
    # normalized so code indentation, Markdown hard breaks, and YAML remain
    # distinct.
    normalized = "\n".join(
        (
            _normalize_newlines(item.content),
            item.source or "",
            item.revision or "",
            "stable" if item.stable else "dynamic",
            item.state_slot or "",
            "" if item.state_version is None else str(item.state_version),
        )
    )
    return hashlib.blake2s(
        normalized.encode("utf-8"),
        digest_size=16,
    ).hexdigest()


def _validate_items(items: Sequence[ContextItem]) -> None:
    seen: set[str] = set()
    for item in items:
        if not item.key:
            raise ValueError("ContextItem.key must not be empty")
        if item.key in seen:
            raise ValueError(f"duplicate ContextItem key: {item.key!r}")
        seen.add(item.key)

        if item.chunk_mode is not None and item.chunk_mode not in CHUNK_MODES:
            raise ValueError(
                f"chunk_mode must be one of {sorted(CHUNK_MODES)} or None"
            )
        if (
            item.chunk_target_tokens is not None
            and item.chunk_target_tokens < 1
        ):
            raise ValueError("chunk_target_tokens must be >= 1")


def _compact_opt_in_blocks(
    items: Sequence[ContextItem],
) -> tuple[list[ContextItem], list[str]]:
    compacted: list[ContextItem] = []
    compacted_keys: list[str] = []

    for item in items:
        if not item.dedup_blocks:
            compacted.append(item)
            continue

        content = _deduplicate_blocks(item.content)
        if content != item.content:
            compacted_keys.append(item.key)
            compacted.append(replace(item, content=content))
        else:
            compacted.append(item)

    return compacted, compacted_keys


def _deduplicate_blocks(content: str) -> str:
    if "\n" not in content:
        return content

    paragraph_blocks = re.split(r"\n\s*\n+", _normalize_newlines(content))
    if len(paragraph_blocks) > 1:
        separator = "\n\n"
        blocks = paragraph_blocks
    else:
        separator = "\n"
        blocks = _normalize_newlines(content).splitlines()

    seen: set[str] = set()
    kept: list[str] = []
    for block in blocks:
        if not block.strip():
            continue
        fingerprint = hashlib.blake2s(
            _normalize_newlines(block).encode("utf-8"),
            digest_size=12,
        ).hexdigest()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        # Preserve leading/internal whitespace. The block transform is opt-in.
        kept.append(block)

    return separator.join(kept)


def _chunk_optional_items(
    items: Sequence[ContextItem],
    *,
    counter: TokenCounter,
    reserved_keys: set[str],
) -> tuple[list[ContextItem], list[str]]:
    expanded: list[ContextItem] = []
    chunked_keys: list[str] = []
    used_keys = set(reserved_keys)

    for item in items:
        if item.required or item.chunk_mode is None:
            expanded.append(item)
            continue

        target = item.chunk_target_tokens or 128
        if counter.count(item.content) <= target:
            expanded.append(item)
            continue

        blocks = _chunk_blocks(item.content, mode=item.chunk_mode)
        groups = _pack_blocks(
            blocks,
            mode=item.chunk_mode,
            target_tokens=target,
            counter=counter,
        )
        if len(groups) <= 1:
            expanded.append(item)
            continue

        chunked_keys.append(item.key)
        for index, chunk_content in enumerate(groups, start=1):
            base = f"{item.key}::chunk:{index:03d}"
            key = base
            collision = 1
            while key in used_keys:
                collision += 1
                key = f"{base}:{collision}"
            used_keys.add(key)
            expanded.append(
                replace(
                    item,
                    key=key,
                    content=chunk_content,
                    chunk_mode=None,
                    chunk_target_tokens=None,
                )
            )

    return expanded, chunked_keys


def _chunk_blocks(content: str, *, mode: str) -> list[str]:
    normalized = _normalize_newlines(content)
    if mode == "line":
        return [line for line in normalized.splitlines() if line.strip()]

    # Paragraph mode intentionally keeps paragraph content intact. A caller
    # choosing this mode asserts that paragraphs are independently discardable.
    return [
        block
        for block in re.split(r"\n\s*\n+", normalized)
        if block.strip()
    ]


def _pack_blocks(
    blocks: Sequence[str],
    *,
    mode: str,
    target_tokens: int,
    counter: TokenCounter,
) -> list[str]:
    if not blocks:
        return []

    separator = "\n" if mode == "line" else "\n\n"
    groups: list[str] = []
    current: list[str] = []

    for block in blocks:
        candidate = separator.join((*current, block))
        if current and counter.count(candidate) > target_tokens:
            groups.append(separator.join(current))
            current = [block]
        else:
            current.append(block)

    if current:
        groups.append(separator.join(current))
    return groups


def _remove_superseded_state(
    items: Sequence[ContextItem],
) -> tuple[list[ContextItem], list[str]]:
    max_version_by_slot: dict[str, int] = {}
    for item in items:
        if item.state_slot is None or item.state_version is None:
            continue
        previous = max_version_by_slot.get(item.state_slot)
        if previous is None or item.state_version > previous:
            max_version_by_slot[item.state_slot] = item.state_version

    kept: list[ContextItem] = []
    superseded: list[str] = []
    for item in items:
        if item.state_slot is None or item.state_version is None:
            kept.append(item)
            continue

        latest = max_version_by_slot[item.state_slot]
        if item.state_version < latest:
            superseded.append(item.key)
        else:
            kept.append(item)

    return kept, superseded


def _selection_key(
    item: ContextItem,
    *,
    strategy: str,
    counter: TokenCounter,
) -> tuple[float, float, int, int, str]:
    tokens = max(1, counter.count(item.content))
    density = item.priority / tokens

    if strategy == "density":
        return (-density, -float(item.priority), -item.priority, tokens, item.key)

    if strategy == "balanced":
        # Preserve coarse priority ordering, then maximize value/token inside
        # the same priority band. This avoids pure density's tendency to let
        # many tiny low-priority items outrank one clearly higher-priority item.
        priority_band = item.priority // 10
        return (
            -float(priority_band),
            -density,
            -item.priority,
            tokens,
            item.key,
        )

    return (
        -float(item.priority),
        -float(item.priority),
        -item.priority,
        tokens,
        item.key,
    )


def _deduplicate(
    items: Sequence[ContextItem],
) -> tuple[list[ContextItem], list[str]]:
    groups: dict[str, list[tuple[int, ContextItem]]] = {}
    order: list[str] = []

    for index, item in enumerate(items):
        fingerprint = _content_fingerprint(item.content)
        if fingerprint not in groups:
            groups[fingerprint] = []
            order.append(fingerprint)
        groups[fingerprint].append((index, item))

    survivors: list[tuple[int, ContextItem]] = []
    duplicate_keys: list[str] = []

    for fingerprint in order:
        group = groups[fingerprint]
        survivor_index, survivor = max(
            group,
            key=lambda pair: _duplicate_rank(pair[1], pair[0]),
        )
        survivors.append((survivor_index, survivor))
        duplicate_keys.extend(
            item.key
            for index, item in group
            if index != survivor_index
        )

    survivors.sort(key=lambda pair: pair[0])
    return [item for _, item in survivors], duplicate_keys


def _duplicate_rank(item: ContextItem, index: int) -> tuple[int, int, int, int]:
    return (
        1 if item.required else 0,
        item.priority,
        1 if item.stable else 0,
        -index,
    )


def _content_fingerprint(content: str) -> str:
    # Do not collapse whitespace here. Code/YAML/Markdown formatting may be
    # semantically meaningful.
    normalized = _normalize_newlines(content)
    return hashlib.blake2s(
        normalized.encode("utf-8"),
        digest_size=12,
    ).hexdigest()


def _normalize_newlines(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n")
