from __future__ import annotations

from typing import Any


PHASES = (
    "dedup",
    "supersession",
    "block-compaction",
    "balanced-packing",
    "chunking-code-safety",
)


def build_baseline_1000_workload(
    *,
    cases: int = 1000,
    noise_items: int = 8,
) -> list[dict[str, Any]]:
    """Build a deterministic baseline-only stress corpus.

    The corpus deliberately exercises safe structural reduction features.
    It never depends on adaptive policy state.
    """

    if cases < len(PHASES):
        raise ValueError(f"cases must be >= {len(PHASES)}")
    if noise_items < 1:
        raise ValueError("noise_items must be >= 1")

    workload: list[dict[str, Any]] = []
    phase_size = max(1, cases // len(PHASES))

    for index in range(1, cases + 1):
        phase_index = min((index - 1) // phase_size, len(PHASES) - 1)
        phase = PHASES[phase_index]
        code = f"BL-{index:04d}"

        if phase == "dedup":
            case = _dedup_case(index, code, noise_items)
        elif phase == "supersession":
            case = _supersession_case(index, code, noise_items)
        elif phase == "block-compaction":
            case = _block_case(index, code, noise_items)
        elif phase == "balanced-packing":
            case = _balanced_case(index, code, noise_items)
        else:
            case = _chunk_code_case(index, code, noise_items)

        case["phase"] = phase
        case["turn"] = index
        workload.append(case)

    return workload


def _base_required() -> dict[str, Any]:
    return {
        "key": "baseline-contract",
        "content": "The baseline benchmark protocol is B1000.",
        "required": True,
        "priority": 100,
        "stable": True,
        "source": "baseline-contract",
        "revision": "1",
    }


def _dedup_case(
    index: int,
    code: str,
    noise_items: int,
) -> dict[str, Any]:
    critical = {
        "key": f"answer-{index:04d}",
        "content": f"Current baseline code is {code}.",
        "priority": 95,
        "source": "current-answer",
    }
    context = [
        _base_required(),
        critical,
        {
            "key": f"answer-copy-{index:04d}",
            "content": critical["content"],
            "priority": 15,
            "source": "duplicate-answer",
        },
    ]
    for n in range(1, noise_items + 1):
        context.append(
            {
                "key": f"dedup-noise-{index:04d}-{n:02d}",
                "content": (
                    f"Archived duplicate-safe note {n}. "
                    "This note does not define the current baseline code."
                ),
                "priority": 5,
                "source": "archive",
            }
        )

    return _case(index, code, context)


def _supersession_case(
    index: int,
    code: str,
    noise_items: int,
) -> dict[str, Any]:
    context = [_base_required()]
    versions = max(4, noise_items)
    for version in range(1, versions + 1):
        is_latest = version == versions
        content = (
            f"Current baseline code is {code}."
            if is_latest
            else f"Obsolete state version {version} must not be used."
        )
        context.append(
            {
                "key": f"state-{index:04d}-v{version:02d}",
                "content": content,
                "priority": 95 if is_latest else 80,
                "source": "operator-state",
                "state_slot": f"task-{index:04d}",
                "state_version": version,
            }
        )

    return _case(index, code, context)


def _block_case(
    index: int,
    code: str,
    noise_items: int,
) -> dict[str, Any]:
    repeated = (
        "Archived telemetry: no current baseline code update occurred.\n"
        "Archived telemetry: no current baseline code update occurred.\n"
        "Archived telemetry: no current baseline code update occurred.\n"
        f"Historical marker for case {index}.\n"
    )
    context = [
        _base_required(),
        {
            "key": f"answer-{index:04d}",
            "content": f"Current baseline code is {code}.",
            "priority": 95,
            "source": "current-answer",
        },
    ]
    for n in range(1, noise_items + 1):
        context.append(
            {
                "key": f"log-{index:04d}-{n:02d}",
                "content": repeated,
                "priority": 8,
                "source": "telemetry-log",
                "dedup_blocks": True,
            }
        )

    return _case(index, code, context)


def _balanced_case(
    index: int,
    code: str,
    noise_items: int,
) -> dict[str, Any]:
    context = [
        _base_required(),
        {
            "key": f"answer-{index:04d}",
            "content": (
                f"Current baseline code is {code}. "
                "This is the highest-value optional fact."
            ),
            "priority": 96,
            "source": "current-answer",
        },
    ]

    for n in range(1, noise_items + 1):
        priority = 70 - (n % 4)
        context.append(
            {
                "key": f"packing-noise-{index:04d}-{n:02d}",
                "content": (
                    f"Medium-value archived note {n} contains operational "
                    "history but no current baseline code."
                ),
                "priority": priority,
                "source": "packing-history",
            }
        )

    return _case(index, code, context)


def _chunk_code_case(
    index: int,
    code: str,
    noise_items: int,
) -> dict[str, Any]:
    paragraphs = []
    for n in range(1, noise_items * 2 + 1):
        paragraphs.append(
            (
                f"Historical paragraph {n} for case {index}. "
                "It is independently discardable and contains no current "
                "baseline answer."
            )
        )

    # These two blocks differ only in indentation. Safe fingerprinting must
    # never collapse them into one item.
    code_a = "if ready:\n    run_primary()"
    code_b = "if ready:\n        run_secondary()"

    context = [
        _base_required(),
        {
            "key": f"answer-{index:04d}",
            "content": f"Current baseline code is {code}.",
            "priority": 95,
            "source": "current-answer",
        },
        {
            "key": f"chunk-history-{index:04d}",
            "content": "\n\n".join(paragraphs),
            "priority": 20,
            "source": "chunkable-history",
            "chunk_mode": "paragraph",
            "chunk_target_tokens": 48,
        },
        {
            "key": f"code-a-{index:04d}",
            "content": code_a,
            "priority": 35,
            "source": "code",
        },
        {
            "key": f"code-b-{index:04d}",
            "content": code_b,
            "priority": 34,
            "source": "code",
        },
    ]

    return _case(index, code, context)


def _case(
    index: int,
    code: str,
    context: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": f"baseline-{index:04d}",
        "group": "baseline-1000",
        "task": (
            "What is the current baseline code? Also state the baseline "
            "benchmark protocol."
        ),
        "expected_contains": [code, "B1000"],
        "context": context,
    }
