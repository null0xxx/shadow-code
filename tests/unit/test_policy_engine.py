from dataclasses import FrozenInstanceError, asdict
from importlib import import_module

import pytest
from pydantic import BaseModel, ConfigDict

from shadow_code.domain import policy as policy_domain
from shadow_code.domain.tools import (
    Capability,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolResult,
    ToolSpec,
    ValidatedToolCall,
)


class EmptyArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


_handler_invocations = 0


def _forbidden_handler(call: ToolCall, arguments: BaseModel, context: object) -> ToolResult:
    global _handler_invocations
    _handler_invocations += 1
    raise AssertionError("policy evaluation must not invoke handlers")


def _validated(
    *,
    capability: Capability = Capability.FILESYSTEM_READ,
    side_effects: SideEffect = SideEffect.NONE,
    risk: RiskLevel = RiskLevel.LOW,
    call_id: str = "call-1",
    name: str = "probe",
) -> ValidatedToolCall:
    call = ToolCall(call_id=call_id, name=name, arguments={})
    spec = ToolSpec(
        name=name,
        version="1",
        description="Classify a policy probe.",
        args_model=EmptyArgs,
        handler=_forbidden_handler,
        capability=capability,
        risk=risk,
        side_effects=side_effects,
        timeout_seconds=1,
        max_output_chars=100,
        idempotency=side_effects is SideEffect.NONE,
        parallel_safety=side_effects is SideEffect.NONE,
        renderer_hint="text",
    )
    return ValidatedToolCall(call=call, spec=spec)


def _engine(facts: object):
    engine_module = import_module("shadow_code.policy.engine")
    return engine_module.PolicyEngine(facts)


_WORKSPACE_IDENTITY = policy_domain.WorkspaceIdentity(device=7, inode=11)


def test_policy_domain_types_are_stable_immutable_and_defensively_frozen() -> None:
    source = {Capability.FILESYSTEM_READ}
    identity = policy_domain.WorkspaceIdentity(device=7, inode=11)
    facts = policy_domain.PolicyFacts(source, identity)
    source.clear()

    assert facts.granted_capabilities == frozenset({Capability.FILESYSTEM_READ})
    assert facts.workspace_identity == identity
    assert policy_domain.PolicyDisposition.ALLOW.value == "allow"
    assert policy_domain.PolicyDisposition.REQUIRE_APPROVAL.value == "require_approval"
    assert policy_domain.PolicyDisposition.DENY.value == "deny"
    assert tuple(reason.value for reason in policy_domain.PolicyReason) == (
        "invalid_call",
        "capability_not_granted",
        "workspace_unavailable",
        "read_only",
        "side_effecting",
        "side_effect_unknown",
        "sensitive_capability",
        "unsupported_policy_case",
    )

    decision = policy_domain.PolicyDecision(
        call_id=None,
        tool_name=None,
        disposition=policy_domain.PolicyDisposition.DENY,
        reason=policy_domain.PolicyReason.INVALID_CALL,
    )
    with pytest.raises(FrozenInstanceError):
        facts.workspace_identity = None
    with pytest.raises(FrozenInstanceError):
        decision.reason = policy_domain.PolicyReason.READ_ONLY
    with pytest.raises(TypeError, match="only Capability values"):
        policy_domain.PolicyFacts({"filesystem.read"}, identity)


@pytest.mark.parametrize(
    "input_request",
    [
        ToolCall(call_id="raw-call", name="probe", arguments={}),
        {"call_id": "dict-call", "name": "probe", "arguments": {}},
        object(),
    ],
)
def test_unvalidated_inputs_are_denied(input_request: object) -> None:
    facts = policy_domain.PolicyFacts({Capability.FILESYSTEM_READ}, _WORKSPACE_IDENTITY)

    decision = _engine(facts).decide(input_request)

    assert decision == policy_domain.PolicyDecision(
        call_id=None,
        tool_name=None,
        disposition=policy_domain.PolicyDisposition.DENY,
        reason=policy_domain.PolicyReason.INVALID_CALL,
    )


@pytest.mark.parametrize(
    ("capabilities", "workspace_identity", "side_effects", "reason"),
    [
        *[
            (
                set(),
                _WORKSPACE_IDENTITY,
                side_effects,
                policy_domain.PolicyReason.CAPABILITY_NOT_GRANTED,
            )
            for side_effects in SideEffect
        ],
        (set(), None, SideEffect.UNKNOWN, policy_domain.PolicyReason.CAPABILITY_NOT_GRANTED),
        (
            {Capability.FILESYSTEM_READ},
            None,
            SideEffect.NONE,
            policy_domain.PolicyReason.WORKSPACE_UNAVAILABLE,
        ),
    ],
)
def test_missing_authority_uses_exact_precedence(
    capabilities: set[Capability],
    workspace_identity: policy_domain.WorkspaceIdentity | None,
    side_effects: SideEffect,
    reason: policy_domain.PolicyReason,
) -> None:
    facts = policy_domain.PolicyFacts(capabilities, workspace_identity)

    decision = _engine(facts).decide(_validated(side_effects=side_effects))

    assert decision.disposition is policy_domain.PolicyDisposition.DENY
    assert decision.reason is reason


def test_unsupported_policy_metadata_denies() -> None:
    facts = policy_domain.PolicyFacts({Capability.FILESYSTEM_READ}, _WORKSPACE_IDENTITY)
    request = _validated()
    unsupported_spec = request.spec.model_copy(update={"side_effects": "future"})
    unsupported_request = request.model_copy(update={"spec": unsupported_spec})

    decision = _engine(facts).decide(unsupported_request)

    assert decision.disposition is policy_domain.PolicyDisposition.DENY
    assert decision.reason is policy_domain.PolicyReason.UNSUPPORTED_POLICY_CASE


_SENSITIVE_CAPABILITIES = (
    Capability.FILESYSTEM_WRITE,
    Capability.PROCESS_EXECUTE,
    Capability.NETWORK_ACCESS,
    Capability.MCP_INVOKE,
)


@pytest.mark.parametrize(
    ("capability", "side_effects", "risk", "disposition", "reason"),
    [
        (
            Capability.FILESYSTEM_READ,
            SideEffect.NONE,
            RiskLevel.HIGH,
            policy_domain.PolicyDisposition.ALLOW,
            policy_domain.PolicyReason.READ_ONLY,
        ),
        *[
            (
                capability,
                SideEffect.NONE,
                RiskLevel.MEDIUM,
                policy_domain.PolicyDisposition.REQUIRE_APPROVAL,
                policy_domain.PolicyReason.SENSITIVE_CAPABILITY,
            )
            for capability in _SENSITIVE_CAPABILITIES
        ],
        *[
            (
                capability,
                SideEffect.MUTATING,
                RiskLevel.LOW,
                policy_domain.PolicyDisposition.REQUIRE_APPROVAL,
                policy_domain.PolicyReason.SIDE_EFFECTING,
            )
            for capability in Capability
        ],
        *[
            (
                capability,
                SideEffect.UNKNOWN,
                RiskLevel.MEDIUM,
                policy_domain.PolicyDisposition.REQUIRE_APPROVAL,
                policy_domain.PolicyReason.SIDE_EFFECT_UNKNOWN,
            )
            for capability in Capability
        ],
    ],
)
def test_exact_policy_matrix(
    capability: Capability,
    side_effects: SideEffect,
    risk: RiskLevel,
    disposition: policy_domain.PolicyDisposition,
    reason: policy_domain.PolicyReason,
) -> None:
    facts = policy_domain.PolicyFacts({capability}, _WORKSPACE_IDENTITY)

    decision = _engine(facts).decide(
        _validated(capability=capability, side_effects=side_effects, risk=risk)
    )

    assert decision.disposition is disposition
    assert decision.reason is reason


def test_valid_decision_is_correlated_deterministic_and_handler_free() -> None:
    global _handler_invocations
    _handler_invocations = 0
    facts = policy_domain.PolicyFacts({Capability.FILESYSTEM_READ}, _WORKSPACE_IDENTITY)
    request = _validated(call_id="correlated-1", name="correlated_probe")
    engine = _engine(facts)

    first = engine.decide(request)
    second = engine.decide(request)

    assert first == second
    assert first.call_id == "correlated-1"
    assert first.tool_name == "correlated_probe"
    assert set(asdict(first)) == {"call_id", "tool_name", "disposition", "reason"}
    assert "arguments" not in repr(first)
    assert _handler_invocations == 0


def test_policy_engine_is_immutable() -> None:
    facts = policy_domain.PolicyFacts({Capability.FILESYSTEM_READ}, _WORKSPACE_IDENTITY)
    engine = _engine(facts)

    with pytest.raises(FrozenInstanceError):
        engine.facts = policy_domain.PolicyFacts(set(), None)


def test_policy_package_exports_engine_and_workspace_guard() -> None:
    import shadow_code.policy as policy_package

    engine_type = import_module("shadow_code.policy.engine").PolicyEngine

    assert policy_package.PolicyEngine is engine_type
    assert policy_package.__all__ == ["PolicyEngine", "WorkspaceGuard"]
