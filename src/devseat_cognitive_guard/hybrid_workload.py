from __future__ import annotations

from typing import Any


FAMILIES = (
    "build-fact",
    "runtime-fact",
    "dependency-fact",
    "checkpoint-fact",
    "config-fact",
)


def build_hybrid_100_workload(
    *,
    cases: int = 100,
    decoys_per_case: int = 9,
) -> list[dict[str, Any]]:
    """Build a four-phase workload for baseline-vs-adaptive comparison.

    Phases:
      1-25: repeated learnable evidence families.
      26-50: source shift; key families remain stable.
      51-75: source collision; useful and useless items share sources.
      76-100: concept drift; an archive family that was previously safe
              becomes task-critical.
    """

    if cases < 20:
        raise ValueError("cases must be >= 20")
    if decoys_per_case < 1:
        raise ValueError("decoys_per_case must be >= 1")

    phase_size = max(1, cases // 4)
    workload: list[dict[str, Any]] = []

    for index in range(1, cases + 1):
        phase = _phase_for(index, phase_size, cases)
        family = FAMILIES[(index - 1) % len(FAMILIES)]
        code = f"HB-{index:03d}"

        context: list[dict[str, Any]] = [
            {
                "key": "hybrid-contract",
                "content": "The benchmark protocol version is H100.",
                "required": True,
                "priority": 100,
                "stable": True,
                "source": "benchmark-contract",
                "revision": "1",
            }
        ]

        for decoy in range(1, decoys_per_case + 1):
            if phase == "collision" and decoy <= 3:
                source = "mixed-observation"
            else:
                source = "hybrid-decoy"

            context.append(
                {
                    "key": f"noise-{index:03d}-{decoy:02d}",
                    "content": (
                        f"Archived decoy {decoy}: this observation is not the "
                        "current hybrid answer and may be omitted."
                    ),
                    "priority": 10,
                    "source": source,
                }
            )

        # For the first three phases, archive-note is consistently safe noise.
        # This intentionally teaches a negative bias that the drift phase must
        # later reverse.
        if phase != "drift":
            context.append(
                {
                    "key": f"archive-note-{index:03d}",
                    "content": (
                        "Historical archive note: no current hybrid answer is "
                        "defined by this record."
                    ),
                    "priority": 1,
                    "source": "archive",
                }
            )

        if phase == "repeat":
            critical_source = "evidence-train"
            critical_key = f"{family}-{index:03d}"
        elif phase == "source-shift":
            critical_source = "evidence-unseen"
            critical_key = f"{family}-{index:03d}"
        elif phase == "collision":
            critical_source = "mixed-observation"
            critical_key = f"{family}-{index:03d}"
        else:
            # Concept drift: the previously safe archive family becomes the
            # only source of the required answer.
            critical_source = "archive"
            critical_key = f"archive-note-{index:03d}"

        context.append(
            {
                "key": critical_key,
                "content": f"Critical hybrid evidence: current hybrid code is {code}.",
                "priority": 1,
                "source": critical_source,
            }
        )

        workload.append(
            {
                "id": f"hybrid-{index:03d}",
                "group": "hybrid-100",
                "phase": phase,
                "turn": index,
                "task": (
                    "What is the current hybrid code? Also state the benchmark "
                    "protocol version."
                ),
                "expected_contains": [code, "H100"],
                "context": context,
            }
        )

    return workload


def _phase_for(index: int, phase_size: int, total: int) -> str:
    if index <= phase_size:
        return "repeat"
    if index <= phase_size * 2:
        return "source-shift"
    if index <= phase_size * 3:
        return "collision"
    if index <= total:
        return "drift"
    raise ValueError("index outside workload")
