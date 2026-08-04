"""One-shot approval authority bound to exact action-plan digests.

An approval is valid for exactly one execution and only while every fact of
the action plan (arguments, preview, workspace identity, tool version,
capability, registry digest) is unchanged. Tokens are never remembered and
never persisted: a consumed or mismatched token is burned.
"""

import hashlib
import json
import secrets

from shadow_code.domain.policy import WorkspaceIdentity
from shadow_code.domain.tools import FrozenModel, ValidatedToolCall


class ActionPlan(FrozenModel):
    """Every fact an approval is bound to, with a deterministic digest."""

    call_id: str
    tool_name: str
    tool_version: str
    capability: str
    canonical_arguments_json: str
    workspace_device: int
    workspace_inode: int
    registry_digest: str
    preview: str

    def digest(self) -> str:
        """SHA-256 hex over the canonical JSON serialization of all fields."""
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode()).hexdigest()


def build_action_plan(
    validated: ValidatedToolCall,
    registry_digest: str,
    workspace: WorkspaceIdentity,
    preview: str,
) -> ActionPlan:
    """Build the action plan for a validated call from pure facts."""
    return ActionPlan(
        call_id=validated.call.call_id,
        tool_name=validated.call.name,
        tool_version=validated.spec.version,
        capability=validated.spec.capability.value,
        canonical_arguments_json=validated.canonical_arguments_json(),
        workspace_device=workspace.device,
        workspace_inode=workspace.inode,
        registry_digest=registry_digest,
        preview=preview,
    )


def render_action_preview(validated: ValidatedToolCall) -> str:
    """Human-readable summary derived from the same facts the digest covers."""
    spec = validated.spec
    return (
        f"{spec.name} v{spec.version} [{spec.capability.value}] "
        f"arguments={validated.canonical_arguments_json()}"
    )


class ApprovalToken(FrozenModel):
    """One-shot grant for the action plan whose digest it carries."""

    token_id: str
    plan_digest: str


class ApprovalAuthority:
    """Session-scoped issuer and verifier of one-shot approval tokens.

    The only mutable component of the approval flow: it tracks outstanding
    token ids in memory. Consuming a token invalidates it on any attempt —
    including a digest mismatch — so a burned token can never be replayed
    and no grant is ever remembered.
    """

    def __init__(self) -> None:
        self._outstanding: dict[str, str] = {}

    def issue(self, plan: ActionPlan) -> ApprovalToken:
        """Issue a fresh one-shot token bound to the plan digest."""
        token = ApprovalToken(token_id=secrets.token_hex(16), plan_digest=plan.digest())
        self._outstanding[token.token_id] = token.plan_digest
        return token

    def consume(self, token: ApprovalToken, plan: ActionPlan) -> bool:
        """Return True exactly once iff the token is outstanding and digest-bound.

        A known token id is burned by any consume attempt, even when the
        plan digest does not match, so failures leak no oracle and tokens
        cannot be retried against a mutated plan.
        """
        expected = self._outstanding.pop(token.token_id, None)
        return expected is not None and expected == plan.digest()
