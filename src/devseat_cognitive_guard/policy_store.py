from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .learning_policy import AdaptiveContextPolicy


POLICY_STORE_SCHEMA = "devseat.context-policy.v1"


@dataclass(frozen=True)
class PolicyNamespace:
    product: str
    profile: str = "default"

    def __post_init__(self) -> None:
        if not self.product.strip():
            raise ValueError("product must not be empty")
        if not self.profile.strip():
            raise ValueError("profile must not be empty")


class FilePolicyStore:
    """Atomic, product-isolated persistence for adaptive policies.

    Paths are derived from hashes rather than raw product/profile names, so
    repository names, slashes, and user-controlled identifiers cannot escape
    the configured store root.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()

    def load(self, namespace: PolicyNamespace) -> AdaptiveContextPolicy:
        path = self.path_for(namespace)
        if not path.exists():
            return AdaptiveContextPolicy()

        payload = json.loads(path.read_text(encoding="utf-8"))
        self._validate_envelope(payload, namespace)
        return AdaptiveContextPolicy.from_dict(payload["policy"])

    def save(
        self,
        namespace: PolicyNamespace,
        policy: AdaptiveContextPolicy,
    ) -> Path:
        path = self.path_for(namespace)
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema": POLICY_STORE_SCHEMA,
            "product": namespace.product,
            "profile": namespace.profile,
            "policy": policy.to_dict(),
        }
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            json.dumps(envelope, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temp.replace(path)
        return path

    def delete(self, namespace: PolicyNamespace) -> bool:
        path = self.path_for(namespace)
        if not path.exists():
            return False
        path.unlink()
        return True

    def path_for(self, namespace: PolicyNamespace) -> Path:
        product_hash = hashlib.sha256(
            namespace.product.encode("utf-8")
        ).hexdigest()
        profile_hash = hashlib.sha256(
            namespace.profile.encode("utf-8")
        ).hexdigest()
        return self.root / product_hash[:24] / f"{profile_hash[:24]}.json"

    @staticmethod
    def _validate_envelope(
        payload: Any,
        namespace: PolicyNamespace,
    ) -> None:
        if not isinstance(payload, dict):
            raise ValueError("policy store document must be an object")
        if payload.get("schema") != POLICY_STORE_SCHEMA:
            raise ValueError("unsupported policy store schema")
        if payload.get("product") != namespace.product:
            raise ValueError("policy product namespace mismatch")
        if payload.get("profile") != namespace.profile:
            raise ValueError("policy profile namespace mismatch")
        if not isinstance(payload.get("policy"), dict):
            raise ValueError("policy payload must be an object")
