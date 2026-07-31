"""Deterministic tool-call policy classification."""

from dataclasses import dataclass

from shadow_code.domain.policy import (
    PolicyDecision,
    PolicyDisposition,
    PolicyFacts,
    PolicyReason,
)
from shadow_code.domain.tools import Capability, SideEffect, ValidatedToolCall

_SENSITIVE_CAPABILITIES = frozenset(
    {
        Capability.FILESYSTEM_WRITE,
        Capability.PROCESS_EXECUTE,
        Capability.NETWORK_ACCESS,
        Capability.MCP_INVOKE,
    }
)


@dataclass(frozen=True, slots=True)
class PolicyEngine:
    """Classify validated calls without executing them."""

    facts: PolicyFacts

    def decide(self, request: object) -> PolicyDecision:
        if not isinstance(request, ValidatedToolCall):
            return PolicyDecision(
                call_id=None,
                tool_name=None,
                disposition=PolicyDisposition.DENY,
                reason=PolicyReason.INVALID_CALL,
            )
        if request.spec.capability not in self.facts.granted_capabilities:
            return self._decision(
                request,
                PolicyDisposition.DENY,
                PolicyReason.CAPABILITY_NOT_GRANTED,
            )
        if self.facts.workspace_identity is None:
            return self._decision(
                request,
                PolicyDisposition.DENY,
                PolicyReason.WORKSPACE_UNAVAILABLE,
            )

        capability = request.spec.capability
        side_effects = request.spec.side_effects
        if capability is Capability.FILESYSTEM_READ and side_effects is SideEffect.NONE:
            disposition = PolicyDisposition.ALLOW
            reason = PolicyReason.READ_ONLY
        elif side_effects is SideEffect.MUTATING:
            disposition = PolicyDisposition.REQUIRE_APPROVAL
            reason = PolicyReason.SIDE_EFFECTING
        elif side_effects is SideEffect.UNKNOWN:
            disposition = PolicyDisposition.REQUIRE_APPROVAL
            reason = PolicyReason.SIDE_EFFECT_UNKNOWN
        elif capability in _SENSITIVE_CAPABILITIES and side_effects is SideEffect.NONE:
            disposition = PolicyDisposition.REQUIRE_APPROVAL
            reason = PolicyReason.SENSITIVE_CAPABILITY
        else:
            disposition = PolicyDisposition.DENY
            reason = PolicyReason.UNSUPPORTED_POLICY_CASE
        return self._decision(request, disposition, reason)

    @staticmethod
    def _decision(
        request: ValidatedToolCall,
        disposition: PolicyDisposition,
        reason: PolicyReason,
    ) -> PolicyDecision:
        return PolicyDecision(
            call_id=request.call.call_id,
            tool_name=request.call.name,
            disposition=disposition,
            reason=reason,
        )
