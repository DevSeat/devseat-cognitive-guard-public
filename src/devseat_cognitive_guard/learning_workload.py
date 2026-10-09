from __future__ import annotations

from typing import Any


def build_learning_challenge_workload(
    *,
    turns: int = 12,
    decoys_per_turn: int = 9,
) -> list[dict[str, Any]]:
    """Build cases where a low-priority optional fact is initially pruned.

    The repeated evidence source gives the adaptive policy a learnable signal.
    With a tight budget, decoys outrank the evidence at first. After one or
    more compression regressions, learned keep-bias should promote the
    evidence and recover quality on later cases.
    """

    if turns < 1:
        raise ValueError("turns must be >= 1")
    if decoys_per_turn < 1:
        raise ValueError("decoys_per_turn must be >= 1")

    cases: list[dict[str, Any]] = []
    for turn in range(1, turns + 1):
        code = f"AC-{turn:03d}"
        context: list[dict[str, Any]] = []

        for index in range(1, decoys_per_turn + 1):
            context.append(
                {
                    "key": f"decoy-{turn:03d}-{index:02d}",
                    "content": (
                        f"Decoy record {index}: archived observation unrelated "
                        "to the adaptive code and safe to omit."
                    ),
                    "priority": 10,
                    "source": "challenge-decoy",
                }
            )

        context.append(
            {
                "key": f"adaptive-evidence-{turn:03d}",
                "content": f"Adaptive evidence: current adaptive code is {code}.",
                "priority": 1,
                "source": "adaptive-evidence",
            }
        )

        cases.append(
            {
                "id": f"learning-challenge-{turn:03d}",
                "group": "learning-challenge",
                "turn": turn,
                "task": "What is the current adaptive code?",
                "expected_contains": [code],
                "context": context,
            }
        )

    return cases



def build_generalization_challenge_workload(
    *,
    turns: int = 12,
    source_switch_turn: int = 7,
    decoys_per_turn: int = 9,
) -> list[dict[str, Any]]:
    """Challenge transfer across an unseen evidence source.

    The critical key family stays stable while the source changes halfway
    through the workload. A useful learner should be able to transfer some
    keep-bias through the key-family feature instead of memorizing one source.
    """

    if turns < 2:
        raise ValueError("turns must be >= 2")
    if not 2 <= source_switch_turn <= turns:
        raise ValueError("source_switch_turn must be between 2 and turns")

    cases: list[dict[str, Any]] = []
    for turn in range(1, turns + 1):
        code = f"GX-{turn:03d}"
        source = (
            "evidence-train"
            if turn < source_switch_turn
            else "evidence-unseen"
        )

        context: list[dict[str, Any]] = []
        for index in range(1, decoys_per_turn + 1):
            context.append(
                {
                    "key": f"general-decoy-{turn:03d}-{index:02d}",
                    "content": (
                        f"General decoy {index}: archived low-value note "
                        "unrelated to the current generalization code."
                    ),
                    "priority": 10,
                    "source": "general-decoy",
                }
            )

        context.append(
            {
                "key": f"critical-evidence-{turn:03d}",
                "content": (
                    f"Critical evidence: current generalization code is {code}."
                ),
                "priority": 1,
                "source": source,
            }
        )

        cases.append(
            {
                "id": f"generalization-challenge-{turn:03d}",
                "group": "generalization-challenge",
                "turn": turn,
                "task": "What is the current generalization code?",
                "expected_contains": [code],
                "context": context,
            }
        )

    return cases
