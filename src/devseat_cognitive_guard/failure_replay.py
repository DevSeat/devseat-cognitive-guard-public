from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .context_models import ContextItem
from .learning_policy import feature_names


@dataclass(frozen=True)
class ReplayAttribution:
    recovered: bool
    attributed_keys: tuple[str, ...]
    probes: int
    candidate_count: int
    interaction_group: bool = False


@dataclass(frozen=True)
class FailureReplayRecord:
    case_id: str
    token_budget: int | None
    selection_strategy: str
    omitted_count: int
    recovered: bool
    attributed_keys: tuple[str, ...]
    attributed_features: tuple[str, ...]
    probes: int
    interaction_group: bool


def attribute_omission_failure(
    omitted_keys: Sequence[str],
    *,
    probe: Callable[[tuple[str, ...]], bool],
    max_probes: int = 8,
) -> ReplayAttribution:
    """Find a small omitted-key set whose restoration recovers quality.

    The algorithm is bounded and conservative:
    1. prove that restoring all omitted items can recover the failure;
    2. bisect the candidate set;
    3. keep a half only when that half alone recovers quality;
    4. stop on interactions or when the probe budget is exhausted.

    It does not assume that one item is always the sole cause.
    """

    unique = tuple(dict.fromkeys(str(key) for key in omitted_keys))
    if not unique or max_probes < 1:
        return ReplayAttribution(
            recovered=False,
            attributed_keys=(),
            probes=0,
            candidate_count=len(unique),
        )

    probes = 1
    if not probe(unique):
        return ReplayAttribution(
            recovered=False,
            attributed_keys=(),
            probes=probes,
            candidate_count=len(unique),
        )

    current = unique
    interaction_group = False

    while len(current) > 1 and probes < max_probes:
        midpoint = (len(current) + 1) // 2
        left = current[:midpoint]
        right = current[midpoint:]

        probes += 1
        if probe(left):
            current = left
            continue

        if not right or probes >= max_probes:
            interaction_group = True
            break

        probes += 1
        if probe(right):
            current = right
            continue

        # Neither half alone recovers the answer, so the failure likely
        # depends on an interaction between multiple omitted items.
        interaction_group = True
        break

    return ReplayAttribution(
        recovered=True,
        attributed_keys=current,
        probes=probes,
        candidate_count=len(unique),
        interaction_group=interaction_group,
    )


def make_replay_record(
    *,
    case_id: str,
    token_budget: int | None,
    selection_strategy: str,
    attribution: ReplayAttribution,
    items: Sequence[ContextItem],
    omitted_keys: Iterable[str],
) -> FailureReplayRecord:
    omitted = set(omitted_keys)
    attributed = set(attribution.attributed_keys)

    features: list[str] = []
    for item in items:
        if item.key not in omitted or item.key not in attributed:
            continue
        features.extend(feature_names(item))

    return FailureReplayRecord(
        case_id=case_id,
        token_budget=token_budget,
        selection_strategy=selection_strategy,
        omitted_count=len(omitted),
        recovered=attribution.recovered,
        attributed_keys=attribution.attributed_keys,
        attributed_features=tuple(dict.fromkeys(features)),
        probes=attribution.probes,
        interaction_group=attribution.interaction_group,
    )


def append_replay_record(
    path: str | Path,
    record: FailureReplayRecord,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(asdict(record), ensure_ascii=False) + "\n"
        )


def summarize_replay_records(
    records: Sequence[FailureReplayRecord],
) -> Mapping[str, int]:
    return {
        "records": len(records),
        "recovered": sum(record.recovered for record in records),
        "unresolved": sum(not record.recovered for record in records),
        "probes": sum(record.probes for record in records),
        "interaction_groups": sum(
            record.interaction_group for record in records
        ),
        "attributed_keys": sum(
            len(record.attributed_keys) for record in records
        ),
    }
