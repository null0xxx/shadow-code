# PolicyEngine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an immutable, deterministic PolicyEngine that auto-admits only granted, workspace-bound, side-effect-free filesystem reads and otherwise returns typed approval or denial decisions.

**Architecture:** Trusted composition creates immutable `PolicyFacts`; `ToolRegistry` remains the only creator of `ValidatedToolCall`; `PolicyEngine.decide(request: object)` applies an exact fail-closed rule order and returns an immutable correlated `PolicyDecision`. The engine classifies metadata only and remains completely separate from approval authority, path containment, handlers, executor dispatch, providers, prompts, UI, and persistence.

**Tech Stack:** Python 3.10+, frozen/slotted dataclasses, Pydantic v2 tool contracts, pytest, Ruff, mypy, Bandit, Gentle AI native bounded review.

## Global Constraints

- Source of truth: `docs/superpowers/specs/2026-07-31-policy-engine-design.md`.
- Work unit: WU-01c.2 is one reviewable behavior, one native-review candidate, and one implementation commit.
- TDD is mandatory: no production change before its focused test is observed failing for the expected missing behavior.
- Only `FILESYSTEM_READ` + `SideEffect.NONE`, with the capability granted and a non-`None` `WorkspaceIdentity`, may return `ALLOW`.
- `FILESYSTEM_WRITE`, `PROCESS_EXECUTE`, `NETWORK_ACCESS`, and `MCP_INVOKE` never auto-admit, even when marked `SideEffect.NONE`.
- `RiskLevel` is audit/presentation metadata only and cannot weaken, bypass, or reorder policy rules.
- Unvalidated inputs return typed `DENY/INVALID_CALL`; they do not raise into orchestration.
- No approval/token, `ActionPlan`, handler invocation, filesystem access, executor, provider, prompt, UI, persistence, budget, sandbox, or production-loop wiring is permitted.
- Do not repair legacy dispatch, static schemas/prompts, the legacy absolute-path read adapter, or unrelated baseline quality/test defects.
- Implementation scope is exactly four paths: `shadow_code/domain/policy.py`, `shadow_code/policy/engine.py`, `shadow_code/policy/__init__.py`, and `tests/unit/test_policy_engine.py`.
- Forecast is 220-320 authored additions plus deletions; hard stop at more than 400 authored changed lines.
- Runtime scenario is N/A because this boundary is pure and has no I/O or execution behavior.
- Conventional commits only; never add `Co-Authored-By` or AI attribution.
- Execution precondition: the specification and this plan are already committed or otherwise outside the candidate; `git status --short` is clean before implementation starts.

---

## File map

| Path | Action | Responsibility |
|---|---|---|
| `shadow_code/domain/policy.py` | Modify | Add immutable policy dispositions, reasons, facts, and decisions beside the existing workspace facts. |
| `shadow_code/policy/engine.py` | Create | Implement the one deterministic policy classifier and sensitive-capability set. |
| `shadow_code/policy/__init__.py` | Modify | Export `PolicyEngine` while preserving `WorkspaceGuard`. |
| `tests/unit/test_policy_engine.py` | Create | Prove immutability, rule precedence, risk invariants, determinism, correlation, and zero handler invocation. |

No other path may be changed. `shadow_code/domain/tools.py`, registry/catalog code, production composition, and the approved specification are read-only dependencies.

### Task 1: Implement the immutable PolicyEngine work unit

**Files:**
- Modify: `shadow_code/domain/policy.py`
- Create: `shadow_code/policy/engine.py`
- Modify: `shadow_code/policy/__init__.py`
- Create: `tests/unit/test_policy_engine.py`

**Interfaces:**
- Consumes: `Capability`, `RiskLevel`, `SideEffect`, `ToolCall`, `ToolResult`, `ToolSpec`, and `ValidatedToolCall` from `shadow_code.domain.tools`.
- Consumes: existing `WorkspaceIdentity(device: int, inode: int)` from `shadow_code.domain.policy`.
- Produces: `PolicyFacts(granted_capabilities: Iterable[Capability], workspace_identity: WorkspaceIdentity | None)` with stored `frozenset[Capability]`.
- Produces: `PolicyDecision(call_id: str | None, tool_name: str | None, disposition: PolicyDisposition, reason: PolicyReason)`.
- Produces: `PolicyEngine(facts: PolicyFacts)` and `PolicyEngine.decide(request: object) -> PolicyDecision`.
- Exports: `shadow_code.policy.PolicyEngine` and `shadow_code.policy.WorkspaceGuard`.

#### Micro-cycle A: immutable domain types and defensive capability freeze

- [ ] **Step 1: Create the test module with real ToolSpec helpers and the failing domain test**

Create `tests/unit/test_policy_engine.py` with this initial content:

```python
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
```

- [ ] **Step 2: Run Micro-cycle A RED and verify the missing type is the cause**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/unit/test_policy_engine.py::test_policy_domain_types_are_stable_immutable_and_defensively_frozen -q
```

Expected: `1 failed`; the failure is `AttributeError` because `shadow_code.domain.policy` has no `PolicyFacts` yet. If collection fails for a typo or the test passes, correct the test before touching production code.

- [ ] **Step 3: Add the minimal immutable policy domain types**

In `shadow_code/domain/policy.py`, add the `Iterable` and `Capability` imports, then add these declarations without changing the existing workspace declarations:

```python
from collections.abc import Iterable

from shadow_code.domain.tools import Capability


class PolicyDisposition(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


class PolicyReason(str, Enum):
    INVALID_CALL = "invalid_call"
    CAPABILITY_NOT_GRANTED = "capability_not_granted"
    WORKSPACE_UNAVAILABLE = "workspace_unavailable"
    READ_ONLY = "read_only"
    SIDE_EFFECTING = "side_effecting"
    SIDE_EFFECT_UNKNOWN = "side_effect_unknown"
    SENSITIVE_CAPABILITY = "sensitive_capability"
    UNSUPPORTED_POLICY_CASE = "unsupported_policy_case"


@dataclass(frozen=True, slots=True, init=False)
class PolicyFacts:
    granted_capabilities: frozenset[Capability]
    workspace_identity: WorkspaceIdentity | None

    def __init__(
        self,
        granted_capabilities: Iterable[Capability],
        workspace_identity: WorkspaceIdentity | None,
    ) -> None:
        frozen_capabilities = frozenset(granted_capabilities)
        if any(not isinstance(capability, Capability) for capability in frozen_capabilities):
            raise TypeError("granted_capabilities must contain only Capability values")
        object.__setattr__(self, "granted_capabilities", frozen_capabilities)
        object.__setattr__(self, "workspace_identity", workspace_identity)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    call_id: str | None
    tool_name: str | None
    disposition: PolicyDisposition
    reason: PolicyReason
```

Place `PolicyFacts` after `WorkspaceIdentity`, because its annotation consumes that type. Place both enums before the dataclasses.

- [ ] **Step 4: Run Micro-cycle A GREEN**

Run the Step 2 command again.

Expected: `1 passed`.

#### Micro-cycle B: invalid input and missing authority deny first

- [ ] **Step 5: Append failing invalid-input and missing-authority tests**

Append to `tests/unit/test_policy_engine.py`:

```python
@pytest.mark.parametrize(
    "request",
    [
        ToolCall(call_id="raw-call", name="probe", arguments={}),
        {"call_id": "dict-call", "name": "probe", "arguments": {}},
        object(),
    ],
)
def test_unvalidated_inputs_are_denied(request: object) -> None:
    facts = policy_domain.PolicyFacts(
        {Capability.FILESYSTEM_READ},
        policy_domain.WorkspaceIdentity(device=7, inode=11),
    )

    decision = _engine(facts).decide(request)

    assert decision == policy_domain.PolicyDecision(
        call_id=None,
        tool_name=None,
        disposition=policy_domain.PolicyDisposition.DENY,
        reason=policy_domain.PolicyReason.INVALID_CALL,
    )


_WORKSPACE_IDENTITY = policy_domain.WorkspaceIdentity(device=7, inode=11)


@pytest.mark.parametrize(
    ("capabilities", "workspace_identity", "side_effects", "reason"),
    [
        *[
            (set(), _WORKSPACE_IDENTITY, side_effects, policy_domain.PolicyReason.CAPABILITY_NOT_GRANTED)
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
    facts = policy_domain.PolicyFacts(
        {Capability.FILESYSTEM_READ},
        policy_domain.WorkspaceIdentity(device=7, inode=11),
    )
    request = _validated()
    unsupported_spec = request.spec.model_copy(update={"side_effects": "future"})
    unsupported_request = request.model_copy(update={"spec": unsupported_spec})

    decision = _engine(facts).decide(unsupported_request)

    assert decision.disposition is policy_domain.PolicyDisposition.DENY
    assert decision.reason is policy_domain.PolicyReason.UNSUPPORTED_POLICY_CASE
```

The `model_copy` bypass is test-only and exists solely to exercise the final fail-closed branch; production callers still receive `ValidatedToolCall` only from `ToolRegistry.validate_call()`.

- [ ] **Step 6: Run Micro-cycle B RED and verify the engine is missing**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/unit/test_policy_engine.py -k 'unvalidated_inputs or missing_authority or unsupported_policy' -q
```

Expected: `9 failed`; each failure is caused by `ModuleNotFoundError: No module named 'shadow_code.policy.engine'` raised inside `_engine()`, not collection or fixture errors.

- [ ] **Step 7: Create the minimal fail-closed engine for rules 1-3 and the default denial**

Create `shadow_code/policy/engine.py`:

```python
"""Deterministic tool-call policy classification."""

from shadow_code.domain.policy import (
    PolicyDecision,
    PolicyDisposition,
    PolicyFacts,
    PolicyReason,
)
from shadow_code.domain.tools import ValidatedToolCall


class PolicyEngine:
    """Classify validated calls without executing them."""

    def __init__(self, facts: PolicyFacts) -> None:
        self.facts = facts

    def decide(self, request: object) -> PolicyDecision:
        if not isinstance(request, ValidatedToolCall):
            return PolicyDecision(
                call_id=None,
                tool_name=None,
                disposition=PolicyDisposition.DENY,
                reason=PolicyReason.INVALID_CALL,
            )
        if request.spec.capability not in self.facts.granted_capabilities:
            return PolicyDecision(
                call_id=None,
                tool_name=None,
                disposition=PolicyDisposition.DENY,
                reason=PolicyReason.CAPABILITY_NOT_GRANTED,
            )
        if self.facts.workspace_identity is None:
            return PolicyDecision(
                call_id=None,
                tool_name=None,
                disposition=PolicyDisposition.DENY,
                reason=PolicyReason.WORKSPACE_UNAVAILABLE,
            )
        return PolicyDecision(
            call_id=None,
            tool_name=None,
            disposition=PolicyDisposition.DENY,
            reason=PolicyReason.UNSUPPORTED_POLICY_CASE,
        )
```

This intentionally remains mutable and leaves valid-call correlation empty; later failing tests will drive those requirements.

- [ ] **Step 8: Run Micro-cycle B GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/unit/test_policy_engine.py -q
```

Expected: `10 passed`.

#### Micro-cycle C: exact allow/approval matrix and risk invariants

- [ ] **Step 9: Append the failing capability, side-effect, and risk matrix**

Append to `tests/unit/test_policy_engine.py`:

```python
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
    facts = policy_domain.PolicyFacts(
        {capability},
        policy_domain.WorkspaceIdentity(device=7, inode=11),
    )

    decision = _engine(facts).decide(
        _validated(capability=capability, side_effects=side_effects, risk=risk)
    )

    assert decision.disposition is disposition
    assert decision.reason is reason
```

- [ ] **Step 10: Run Micro-cycle C RED and verify the default denial is exposed**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/unit/test_policy_engine.py -k exact_policy_matrix -q
```

Expected: `15 failed`; actual decisions are `DENY/UNSUPPORTED_POLICY_CASE`, proving the matrix branches do not yet exist. The matrix also proves that high risk cannot block the exact read allow rule and low risk cannot weaken mutation approval.

- [ ] **Step 11: Implement the exact positive allowlist and approval branches**

In `shadow_code/policy/engine.py`, add imports and the sensitive set:

```python
from shadow_code.domain.tools import Capability, SideEffect, ValidatedToolCall

_SENSITIVE_CAPABILITIES = frozenset(
    {
        Capability.FILESYSTEM_WRITE,
        Capability.PROCESS_EXECUTE,
        Capability.NETWORK_ACCESS,
        Capability.MCP_INVOKE,
    }
)
```

Replace only the final default return in `decide()` with:

```python
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
        return PolicyDecision(
            call_id=None,
            tool_name=None,
            disposition=disposition,
            reason=reason,
        )
```

Do not read `request.spec.risk`; the test proves that risk cannot alter authority.

- [ ] **Step 12: Run Micro-cycle C GREEN**

Run the complete focused file:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/unit/test_policy_engine.py -q
```

Expected: `25 passed`.

#### Micro-cycle D: immutable engine, deterministic correlation, and no-handler proof

- [ ] **Step 13: Append failing purity, correlation, and engine-immutability tests**

Append to `tests/unit/test_policy_engine.py`:

```python
def test_valid_decision_is_correlated_deterministic_and_handler_free() -> None:
    global _handler_invocations
    _handler_invocations = 0
    facts = policy_domain.PolicyFacts(
        {Capability.FILESYSTEM_READ},
        policy_domain.WorkspaceIdentity(device=7, inode=11),
    )
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
    facts = policy_domain.PolicyFacts(
        {Capability.FILESYSTEM_READ},
        policy_domain.WorkspaceIdentity(device=7, inode=11),
    )
    engine = _engine(facts)

    with pytest.raises(FrozenInstanceError):
        engine.facts = policy_domain.PolicyFacts(set(), None)
```

- [ ] **Step 14: Run Micro-cycle D RED and verify both missing guarantees**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/unit/test_policy_engine.py -k 'correlated_deterministic or engine_is_immutable' -q
```

Expected: `2 failed`; correlation currently returns `None`, and assignment to mutable `PolicyEngine.facts` does not raise `FrozenInstanceError`.

- [ ] **Step 15: Replace the temporary engine with the final immutable implementation**

Replace `shadow_code/policy/engine.py` completely with:

```python
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
```

This is the final rule order. Do not introduce a generic rule registry, approval service, logging callback, or handler check.

- [ ] **Step 16: Run Micro-cycle D GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/unit/test_policy_engine.py -q
```

Expected: `27 passed`.

#### Micro-cycle E: package exports

- [ ] **Step 17: Append the failing package-export contract**

Append to `tests/unit/test_policy_engine.py`:

```python
def test_policy_package_exports_engine_and_workspace_guard() -> None:
    import shadow_code.policy as policy_package

    engine_type = import_module("shadow_code.policy.engine").PolicyEngine

    assert policy_package.PolicyEngine is engine_type
    assert policy_package.__all__ == ["PolicyEngine", "WorkspaceGuard"]
```

- [ ] **Step 18: Run Micro-cycle E RED and verify the package boundary is missing**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/unit/test_policy_engine.py::test_policy_package_exports_engine_and_workspace_guard -q
```

Expected: `1 failed` with `AttributeError: module 'shadow_code.policy' has no attribute 'PolicyEngine'`.

- [ ] **Step 19: Export the engine without exporting approval or executor symbols**

Replace `shadow_code/policy/__init__.py` with:

```python
"""Code-enforced execution policy adapters."""

from .engine import PolicyEngine
from .workspace import WorkspaceGuard

__all__ = ["PolicyEngine", "WorkspaceGuard"]
```

- [ ] **Step 20: Run Micro-cycle E GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/unit/test_policy_engine.py -q
```

Expected: `28 passed`.

#### Candidate normalization and check-only verification

- [ ] **Step 21: Confirm exact scope before normalization**

Run:

```bash
git status --short
```

Expected paths only:

```text
 M shadow_code/domain/policy.py
 M shadow_code/policy/__init__.py
?? shadow_code/policy/engine.py
?? tests/unit/test_policy_engine.py
```

If the plan/specification is untracked or any unrelated path appears, stop before normalization and return ownership to the orchestrator; do not hide, stage, delete, or absorb it.

- [ ] **Step 22: Measure the authored changed-line budget without mutating the index**

Run:

```bash
git diff --numstat -- shadow_code/domain/policy.py shadow_code/policy/__init__.py
wc -l shadow_code/policy/engine.py tests/unit/test_policy_engine.py
```

Count additions plus deletions from the tracked diff, then add every line in the two new files. Expected total: 220-320. Hard result: no more than 400. If the total is greater than 400, stop and simplify within the same rollback boundary; do not start approval/executor work and do not request a size exception.

- [ ] **Step 23: Run the only source-mutating normalizer before candidate freeze**

Run exactly once:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff format shadow_code/domain/policy.py shadow_code/policy/engine.py shadow_code/policy/__init__.py tests/unit/test_policy_engine.py
```

After this step, every formatter invocation must use `--check`. If any later command changes bytes, paths, or modes, discard the review receipt, normalize again, and start a new review candidate.

- [ ] **Step 24: Run the final focused policy suite after normalization**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/unit/test_policy_engine.py -q
```

Expected: `28 passed`.

- [ ] **Step 25: Run the typed-foundation dependency regression subset**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider tests/unit/test_tool_spec.py tests/unit/test_tool_registry.py tests/unit/test_tool_catalog.py tests/unit/test_workspace_guard.py tests/unit/test_policy_engine.py -q
```

Expected: `81 passed`. Any new failure in these files is candidate-causal and must be fixed before review; do not repair the three known base-only `tests/test_skills.py` failures.

- [ ] **Step 26: Run scoped Ruff checks without mutation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff format --check --no-cache shadow_code/domain/policy.py shadow_code/policy/engine.py shadow_code/policy/__init__.py tests/unit/test_policy_engine.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check --no-cache shadow_code/domain/policy.py shadow_code/policy/engine.py shadow_code/policy/__init__.py tests/unit/test_policy_engine.py
```

Expected: `4 files already formatted` and `All checks passed`.

- [ ] **Step 27: Run scoped mypy, Bandit, and Git hygiene checks**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m mypy --cache-dir /tmp/shadow-wu01c2-mypy-cache shadow_code/domain/policy.py shadow_code/policy/engine.py shadow_code/policy/__init__.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m bandit -c pyproject.toml shadow_code/domain/policy.py shadow_code/policy/engine.py shadow_code/policy/__init__.py
git diff --check -- shadow_code/domain/policy.py shadow_code/policy/__init__.py
```

Expected: mypy reports success for 3 source files; Bandit reports no issues; Git reports no whitespace error. Also inspect the two new files with `rg -n '[[:blank:]]+$' shadow_code/policy/engine.py tests/unit/test_policy_engine.py`; expected: no output.

- [ ] **Step 28: Record the proportional runtime and rollback evidence**

Record these exact facts in the implementation handoff and native final evidence:

```text
Runtime scenario: N/A — PolicyEngine is a pure in-process classifier with no I/O, provider, UI, filesystem-open, approval, handler, or executor boundary.
Rollback: remove shadow_code/policy/engine.py and tests/unit/test_policy_engine.py; revert only the new policy domain types in shadow_code/domain/policy.py and PolicyEngine export in shadow_code/policy/__init__.py. WorkspaceGuard and typed tool foundations remain intact.
```

#### Native review freeze, receipt, and one final commit

- [ ] **Step 29: Freeze the exact candidate with native review START**

First rerun `git status --short` and confirm the exact four-path manifest from Step 21. Then run:

```bash
gentle-ai review start --cwd /home/null/Documents/shadow-code-main --projection workspace > /tmp/shadow-wu01c2-review-start.json
jq '{lineage_id,state,risk_level,selected_lenses,changed_files,changed_lines,correction_budget,target_identity,lens_bindings}' /tmp/shadow-wu01c2-review-start.json
```

Treat this START response as immutable authority. If native output supplies a negotiated consent transition or envelope, relay its complete choice envelope losslessly and run only the exact native follow-up it names; do not invent `--target`. Before each selected lens, supply the exact frozen diff and manifest from START. Do not run any source-mutating command after START.

- [ ] **Step 30: Complete each native-selected review lens once and finalize authority**

For every `lens_bindings` entry, in native order:

1. Delegate one fresh read-only reviewer bound with the exact `GENTLE_AI_REVIEW_BINDING` object from START.
2. Require completed inspection of every manifest path, exact `subject_hash`, ordered `inspection.paths`, findings, and evidence.
3. Save the reviewer result under `/tmp` and pass that exact file to `gentle-ai review capture-result`.
4. After all selected results are captured, pass the result artifact files to `gentle-ai review finalize` in lens order.

If finalize requests the one bounded correction, stop this plan at the correction gate and follow the native correction/validator transaction; do not silently edit reviewed bytes or open a second ordinary correction. Proceed only when native authority produces an approved receipt for the unchanged candidate.

- [ ] **Step 31: Stage the reviewed paths and validate the pre-commit gate**

After native allow, stage only the frozen implementation paths:

```bash
git add -- shadow_code/domain/policy.py shadow_code/policy/engine.py shadow_code/policy/__init__.py tests/unit/test_policy_engine.py
gentle-ai review validate --gate pre-commit --cwd /home/null/Documents/shadow-code-main
```

Expected: native validation allows the same receipt and the staged tree/path set exactly matches the reviewed candidate. Any byte, mode, path, or staged-tree mismatch invalidates the gate; do not commit.

- [ ] **Step 32: Create the single conventional implementation commit and verify repository state**

Run:

```bash
git commit -m "feat(security): add immutable policy engine"
git status --short
git log -1 --format='%h %s%n%b'
```

Expected: commit succeeds, `git status --short` is empty, and the log contains the exact conventional subject with no `Co-Authored-By` or AI attribution. Do not amend, create an intermediate commit, or include unrelated paths.

## Deferred/base-only guardrail

The following are explicitly outside this task and must remain untouched even if repository-wide commands report them:

- `tests/test_skills.py::TestSkillRegistry::test_register_custom_skill`
- `tests/test_skills.py::TestSkillRegistry::test_register_overwrites`
- `tests/test_skills.py::TestBuiltinSkillContent::test_review_skill_mentions_security`
- 25 repository-wide Ruff findings and 10 outside-scope files requiring format normalization
- `shadow_code/repl.py:198` mypy `var-annotated`
- low-severity Bandit B101 at `shadow_code/tools/get_language_rules.py:75`
- `make verify`, because its coverage target writes `.coverage` and is not check-only
- legacy direct dispatcher, static production schemas/prompts, absolute-path read adapter, provider normalization, approvals, executor, persistence, budgets, UI, and prompt activation

## Final acceptance checklist

- [ ] All five RED observations failed for the expected missing behavior before their production changes.
- [ ] Final focused result is exactly 28 passing policy tests.
- [ ] Typed-foundation dependency subset is exactly 81 passing tests.
- [ ] Only granted, workspace-bound, side-effect-free filesystem reads auto-admit.
- [ ] Missing capability denial wins before workspace and approval classification.
- [ ] Missing workspace denies.
- [ ] Mutating and unknown calls require approval after authority checks.
- [ ] Sensitive capabilities never auto-admit when marked `SideEffect.NONE`.
- [ ] `RiskLevel` is not read by the engine and cannot weaken policy.
- [ ] Valid decisions preserve call ID and tool name without arguments or diagnostics.
- [ ] Policy facts, decisions, and engine are immutable and deterministic.
- [ ] Handler invocation count remains zero.
- [ ] No approval, token, executor, filesystem, provider, prompt, UI, persistence, or production wiring was introduced.
- [ ] Runtime is recorded as N/A with the exact reason from Step 28.
- [ ] Authored changed lines are at most 400 and exact scope is four implementation paths.
- [ ] Mutating normalization happened before review START; all later checks were check-only.
- [ ] Native review approved the exact staged candidate and pre-commit gate allowed it.
- [ ] Exactly one implementation commit exists with subject `feat(security): add immutable policy engine` and no attribution trailer.
- [ ] Rollback removes only WU-01c.2 while preserving WorkspaceGuard and typed tool foundations.
