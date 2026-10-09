from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def build_long_workload(
    *,
    turns: int = 12,
    noise_blocks_per_turn: int = 4,
    noise_chars: int = 320,
) -> list[dict[str, Any]]:
    """Materialize the accumulated long-task workload.

    For very large runs, prefer :func:`iter_long_workload` so only one turn's
    context list is materialized at a time.
    """

    return list(
        iter_long_workload(
            turns=turns,
            noise_blocks_per_turn=noise_blocks_per_turn,
            noise_chars=noise_chars,
        )
    )


def iter_long_workload(
    *,
    turns: int = 12,
    noise_blocks_per_turn: int = 4,
    noise_chars: int = 320,
) -> Iterator[dict[str, Any]]:
    """Yield one increasingly accumulated engineering turn at a time.

    Turn N contains all historical low-priority logs and all prior versioned
    engineering states. The deterministic optimizer must keep the current
    state/contract while dropping or superseding accumulated history.

    The generator is intended for 1000+ turn stress tests: once the caller has
    processed a yielded case, that turn's temporary context list can be
    released instead of retaining all 1000 growing context snapshots.
    """

    if turns < 1:
        raise ValueError("turns must be >= 1")
    if noise_blocks_per_turn < 0:
        raise ValueError("noise_blocks_per_turn must be >= 0")
    if noise_chars < 32:
        raise ValueError("noise_chars must be >= 32")

    accumulated_noise: list[dict[str, Any]] = []
    previous_states: list[dict[str, Any]] = []

    repeated_rules = [
        {
            "key": f"public-api-rule-{copy}",
            "content": "Do not change public APIs.",
            "required": copy == 0,
            "priority": 100 if copy == 0 else 20,
            "stable": True,
            "source": "repo-rules",
            "revision": "1",
        }
        for copy in range(3)
    ]
    service_contract = {
        "key": "service-contract",
        "content": (
            "The service contract requires telemetry port 4317 and "
            "this contract remains active for the whole task."
        ),
        "required": True,
        "priority": 100,
        "stable": True,
        "source": "contract",
        "revision": "7",
    }

    for turn in range(1, turns + 1):
        checkpoint = f"checkpoint-{turn:04d}"
        module = _module_for(turn)

        for block in range(1, noise_blocks_per_turn + 1):
            accumulated_noise.append(
                {
                    "key": f"log-{turn:04d}-{block:02d}",
                    "content": _noise_text(
                        turn=turn,
                        block=block,
                        target_chars=noise_chars,
                    ),
                    "priority": 1 + ((turn + block) % 8),
                    "stable": False,
                    "source": "historical-log",
                    "dedup_blocks": True,
                }
            )

        old_states = [
            {
                **state,
                "required": False,
                "priority": 15,
                "stable": False,
                "state_slot": "engineering-state",
            }
            for state in previous_states
        ]
        current_state = {
            "key": f"current-state-{turn:04d}",
            "content": (
                f"Current engineering state: target module is {module}; "
                f"active checkpoint is {checkpoint}; "
                f"the current operation is turn {turn}."
            ),
            "required": True,
            "priority": 100,
            "stable": False,
            "source": "operator-state",
            "revision": str(turn),
            "state_slot": "engineering-state",
            "state_version": turn,
        }

        context = [
            *repeated_rules,
            service_contract,
            *old_states,
            current_state,
            *accumulated_noise,
        ]

        yield {
            "id": f"long-turn-{turn:04d}",
            "group": "long-task",
            "phase": "long-task",
            "turn": turn,
            "task": (
                "Report the current target module and active checkpoint. "
                "Also state the required telemetry port."
            ),
            "expected_contains": [module, checkpoint, "4317"],
            "expected_not_contains": [
                f"checkpoint-{old_turn:04d}"
                for old_turn in range(1, turn)
            ],
            "context": context,
        }
        previous_states.append(current_state)


def _module_for(turn: int) -> str:
    modules = (
        "parser.py",
        "scheduler.py",
        "context_store.py",
        "worker.py",
        "router.py",
        "validator.py",
    )
    return modules[(turn - 1) % len(modules)]


def _noise_text(*, turn: int, block: int, target_chars: int) -> str:
    header = (
        f"Historical diagnostic log from turn {turn}, block {block}. "
        "The archived operation completed without changing the current target."
    )
    repeated = (
        "Archived telemetry: no active checkpoint update occurred.\n"
        "Archived telemetry: no active checkpoint update occurred.\n"
    )
    tail = (
        "This record is low-priority history and does not define the active "
        "checkpoint, target module, or telemetry contract."
    )
    seed = f"{header}\n{repeated}{tail}\n"
    repeats = (target_chars // len(seed)) + 1
    return (seed * repeats)[:target_chars]


_COMPLEX_MODULES = (
    "planner.py",
    "executor.py",
    "context_store.py",
    "policy_engine.py",
    "router.py",
    "validator.py",
    "snapshotter.py",
    "recovery.py",
)

_COMPLEX_RINGS = ("canary", "beta", "stable", "shadow")
_COMPLEX_RISKS = ("low", "medium", "high", "critical")

_COMPLEX_LANES = {
    ("canary", "low"): "lane-c1",
    ("canary", "medium"): "lane-c1",
    ("canary", "high"): "lane-c2",
    ("canary", "critical"): "lane-c2",
    ("beta", "low"): "lane-b1",
    ("beta", "medium"): "lane-b1",
    ("beta", "high"): "lane-b2",
    ("beta", "critical"): "lane-b2",
    ("stable", "low"): "lane-s1",
    ("stable", "medium"): "lane-s1",
    ("stable", "high"): "lane-s2",
    ("stable", "critical"): "lane-s2",
    ("shadow", "low"): "lane-x1",
    ("shadow", "medium"): "lane-x1",
    ("shadow", "high"): "lane-x2",
    ("shadow", "critical"): "lane-x2",
}

_COMPLEX_GATES = {
    "low": "auto-approve",
    "medium": "peer-review",
    "high": "dual-review",
    "critical": "security-board",
}


def build_complex_long_workload(
    *,
    turns: int = 50,
    noise_blocks_per_turn: int = 6,
    noise_chars: int = 420,
) -> list[dict[str, Any]]:
    """Materialize the harder cumulative operator workload."""

    return list(
        iter_complex_long_workload(
            turns=turns,
            noise_blocks_per_turn=noise_blocks_per_turn,
            noise_chars=noise_chars,
        )
    )


def iter_complex_long_workload(
    *,
    turns: int = 50,
    noise_blocks_per_turn: int = 6,
    noise_chars: int = 420,
) -> Iterator[dict[str, Any]]:
    """Yield a cumulative multi-state workload with deterministic reasoning.

    Each turn updates seven independent state slots while retaining all previous
    versions and growing historical noise. The answer also requires deriving a
    rollout lane from deployment-ring/risk policy and an approval gate from
    the current risk level.
    """

    if turns < 1:
        raise ValueError("turns must be >= 1")
    if noise_blocks_per_turn < 0:
        raise ValueError("noise_blocks_per_turn must be >= 0")
    if noise_chars < 64:
        raise ValueError("noise_chars must be >= 64")

    histories: dict[str, list[dict[str, Any]]] = {
        "module": [],
        "checkpoint": [],
        "branch": [],
        "schema": [],
        "deployment": [],
        "dependency": [],
        "risk": [],
    }
    accumulated_noise: list[dict[str, Any]] = []

    invariant_contract = {
        "key": "complex-invariant-contract",
        "content": (
            "Active production contract: telemetry port is 4317; request "
            "timeout is 12s; public API signatures must remain unchanged."
        ),
        "required": True,
        "priority": 100,
        "stable": True,
        "source": "production-contract",
        "revision": "9",
    }
    rollout_policy = {
        "key": "complex-rollout-policy",
        "content": (
            "Rollout lane matrix: canary low/medium -> lane-c1, canary "
            "high/critical -> lane-c2; beta low/medium -> lane-b1, beta "
            "high/critical -> lane-b2; stable low/medium -> lane-s1, stable "
            "high/critical -> lane-s2; shadow low/medium -> lane-x1, shadow "
            "high/critical -> lane-x2."
        ),
        "required": True,
        "priority": 100,
        "stable": True,
        "source": "rollout-policy",
        "revision": "4",
    }
    approval_policy = {
        "key": "complex-approval-policy",
        "content": (
            "Approval policy by current risk: low -> auto-approve; medium -> "
            "peer-review; high -> dual-review; critical -> security-board."
        ),
        "required": True,
        "priority": 100,
        "stable": True,
        "source": "approval-policy",
        "revision": "3",
    }

    for turn in range(1, turns + 1):
        module = _COMPLEX_MODULES[(turn - 1) % len(_COMPLEX_MODULES)]
        checkpoint = f"decision-{turn:04d}"
        branch = f"feature/guard-{turn:03d}"
        schema = f"schema-v{100 + turn}"
        ring = _COMPLEX_RINGS[(turn - 1) % len(_COMPLEX_RINGS)]
        risk = _COMPLEX_RISKS[((turn - 1) * 3) % len(_COMPLEX_RISKS)]
        upstream = f"svc-{turn:03d}.internal"
        lane = _COMPLEX_LANES[(ring, risk)]
        gate = _COMPLEX_GATES[risk]

        current_values = {
            "module": (
                f"Current target module is {module}.",
                module,
            ),
            "checkpoint": (
                f"Current decision checkpoint is {checkpoint}.",
                checkpoint,
            ),
            "branch": (
                f"Current working branch is {branch}.",
                branch,
            ),
            "schema": (
                f"Current schema revision is {schema}.",
                schema,
            ),
            "deployment": (
                f"Current deployment ring is {ring}.",
                ring,
            ),
            "dependency": (
                f"Current upstream dependency is {upstream}.",
                upstream,
            ),
            "risk": (
                f"Current risk class is {risk}.",
                risk,
            ),
        }

        old_states: list[dict[str, Any]] = []
        current_states: list[dict[str, Any]] = []
        for slot, (content, _) in current_values.items():
            old_states.extend(
                {
                    **state,
                    "required": False,
                    "priority": 14,
                    "stable": False,
                }
                for state in histories[slot]
            )
            item = {
                "key": f"complex-{slot}-{turn:04d}",
                "content": content,
                "required": True,
                "priority": 100,
                "stable": False,
                "source": "operator-state",
                "revision": str(turn),
                "state_slot": f"complex-{slot}",
                "state_version": turn,
            }
            current_states.append(item)
            histories[slot].append(item)

        prior_checkpoint = (
            f"decision-{turn - 1:04d}" if turn > 1 else "decision-0000"
        )
        prior_branch = (
            f"feature/guard-{turn - 1:03d}"
            if turn > 1
            else "feature/guard-000"
        )
        prior_schema = (
            f"schema-v{99 + turn}" if turn > 1 else "schema-v100"
        )

        for block in range(1, noise_blocks_per_turn + 1):
            accumulated_noise.append(
                {
                    "key": f"complex-log-{turn:04d}-{block:02d}",
                    "content": _complex_noise_text(
                        turn=turn,
                        block=block,
                        target_chars=noise_chars,
                        prior_checkpoint=prior_checkpoint,
                        prior_branch=prior_branch,
                        prior_schema=prior_schema,
                    ),
                    "priority": 1 + ((turn + block) % 7),
                    "stable": False,
                    "source": "historical-log",
                    "dedup_blocks": True,
                }
            )

        expected = [
            module,
            checkpoint,
            branch,
            schema,
            ring,
            upstream,
            risk,
            lane,
            gate,
            "4317",
            "12s",
        ]

        forbidden = [
            f"decision-{old_turn:04d}"
            for old_turn in range(1, turn)
        ]
        forbidden.extend(
            f"feature/guard-{old_turn:03d}"
            for old_turn in range(1, turn)
        )
        forbidden.extend(
            f"schema-v{100 + old_turn}"
            for old_turn in range(1, turn)
        )

        yield {
            "id": f"complex-turn-{turn:04d}",
            "group": "complex-long-task",
            "phase": "complex-long-task",
            "turn": turn,
            "task": (
                "Produce the current deployment decision. Report the target "
                "module, decision checkpoint, working branch, schema revision, "
                "deployment ring, upstream dependency, and risk class. Then "
                "derive the rollout lane from the rollout matrix and the "
                "approval gate from the approval policy. Also include the "
                "required telemetry port and request timeout. Do not report "
                "superseded checkpoint, branch, or schema values."
            ),
            "expected_contains": expected,
            "expected_not_contains": forbidden,
            "context": [
                invariant_contract,
                rollout_policy,
                approval_policy,
                *old_states,
                *current_states,
                *accumulated_noise,
            ],
        }


def _complex_noise_text(
    *,
    turn: int,
    block: int,
    target_chars: int,
    prior_checkpoint: str,
    prior_branch: str,
    prior_schema: str,
) -> str:
    seed = (
        f"Archived operator note from turn {turn}, block {block}. "
        f"Historical values mentioned {prior_checkpoint}, {prior_branch}, and "
        f"{prior_schema}; these are archival and must not override current "
        "operator state. A previous reviewer discussed telemetry dashboards "
        "and rollout timing, but this note does not define the active policy, "
        "dependency, branch, schema, checkpoint, deployment ring, or risk.\n"
    )
    repeats = (target_chars // len(seed)) + 1
    return (seed * repeats)[:target_chars]
