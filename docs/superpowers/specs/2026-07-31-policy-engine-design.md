# Admit Only Explicitly Safe Tool Calls

WU-01c.2 adds a small, code-enforced `PolicyEngine` that classifies an already validated tool call as `allow`, `require_approval`, or `deny`. Only a granted, workspace-bound, side-effect-free filesystem read may auto-admit. The engine is immutable, deterministic, and unable to execute tools or create approval authority.

## Review path

1. Verify the rule precedence in [Deterministic decision rules](#deterministic-decision-rules).
2. Confirm the boundaries in [Data flow and ownership](#data-flow-and-ownership).
3. Check the implementation and tests against [Acceptance checklist](#acceptance-checklist).

## Approved context

| Context | Approved decision |
|---|---|
| Product | Private, single-user, Linux/Ollama-first coding agent. |
| Authority | Execution authority is enforced by code and is independent of editable prompts. |
| Workspace | The session workspace is fixed and represented by the committed immutable `WorkspaceIdentity`. |
| Validation | `ToolRegistry.validate_call()` is the only source of `ValidatedToolCall`. |
| Side effects | Every side-effecting or side-effect-unknown call ultimately requires exact-action, one-shot approval. |
| Scope | This work unit implements policy facts and classification only. |
| Dependency | WU-01c.1 `WorkspaceGuard` is committed as `83f5de7`. |

## Goals

- Make the policy result a pure function of immutable session facts and an immutable validated call.
- Auto-admit only the exact low-authority read case.
- Deny missing authority and malformed input before any approval classification.
- Return stable, typed outcomes and reason codes suitable for later audit events.
- Keep the unit independently testable and reversible.

## Non-goals

This work unit does **not**:

- create, consume, persist, or replay approvals or tokens;
- define `ActionPlan`, previews, digests, or approval UI;
- invoke a `ToolSpec.handler` or add executor dispatch;
- inspect paths, open files, or duplicate `WorkspaceGuard` containment;
- activate the typed registry in Ollama, prompts, or the production loop;
- add provider normalization, persistence, event logging, budgets, sandboxing, or shell execution;
- migrate `read_file` from its legacy absolute-path adapter;
- add remembered grants or policy-wide auto-accept.

## Immutable domain types

Add the following to `shadow_code/domain/policy.py` as frozen, slotted domain values. `PolicyFacts` must defensively freeze its capability collection and reject values that are not `Capability` members.

| Type | Required members | Purpose |
|---|---|---|
| `PolicyDisposition(str, Enum)` | `ALLOW = "allow"`, `REQUIRE_APPROVAL = "require_approval"`, `DENY = "deny"` | Stable policy outcome. |
| `PolicyReason(str, Enum)` | The exact members and values listed below | Stable, non-sensitive explanation code. |
| `PolicyFacts` | `granted_capabilities: frozenset[Capability]`; `workspace_identity: WorkspaceIdentity | None` | Immutable session authority supplied by trusted composition code. |
| `PolicyDecision` | `call_id: str | None`; `tool_name: str | None`; `disposition: PolicyDisposition`; `reason: PolicyReason` | Correlated classification without arguments or execution authority. |

For an unvalidated input, `call_id` and `tool_name` are `None`. For a `ValidatedToolCall`, the engine copies both values from its immutable `ToolCall`; callers cannot supply alternative correlation values.

`PolicyDecision` must not contain validated arguments, a handler, preview data, prompt text, approval state, or free-form error text.

`PolicyReason` values are fixed as follows:

- `INVALID_CALL = "invalid_call"`
- `CAPABILITY_NOT_GRANTED = "capability_not_granted"`
- `WORKSPACE_UNAVAILABLE = "workspace_unavailable"`
- `READ_ONLY = "read_only"`
- `SIDE_EFFECTING = "side_effecting"`
- `SIDE_EFFECT_UNKNOWN = "side_effect_unknown"`
- `SENSITIVE_CAPABILITY = "sensitive_capability"`
- `UNSUPPORTED_POLICY_CASE = "unsupported_policy_case"`

## PolicyEngine API

Add `shadow_code/policy/engine.py` with one immutable engine:

```python
engine = PolicyEngine(policy_facts)
decision = engine.decide(request)
```

Contract:

```python
PolicyEngine(facts: PolicyFacts)
PolicyEngine.decide(request: object) -> PolicyDecision
```

- Construction fixes the session facts for the engine lifetime.
- `request: object` is deliberate: runtime type checking must deny unvalidated input rather than trusting annotations.
- `decide()` reads metadata only from `ValidatedToolCall.call` and `ValidatedToolCall.spec`.
- `decide()` has no I/O, mutation, cache, clock, randomness, environment access, or handler invocation.
- Equal facts and equal validated calls produce equal decisions.

## Deterministic decision rules

Evaluate rules in this exact order. The first matching rule is terminal.

| Precedence | Condition | Disposition | Reason |
|---:|---|---|---|
| 1 | `request` is not a `ValidatedToolCall` | `DENY` | `INVALID_CALL` |
| 2 | `request.spec.capability` is absent from `granted_capabilities` | `DENY` | `CAPABILITY_NOT_GRANTED` |
| 3 | `workspace_identity is None` | `DENY` | `WORKSPACE_UNAVAILABLE` |
| 4 | Capability is `FILESYSTEM_READ` and side effects are `NONE` | `ALLOW` | `READ_ONLY` |
| 5 | Side effects are `MUTATING` | `REQUIRE_APPROVAL` | `SIDE_EFFECTING` |
| 6 | Side effects are `UNKNOWN` | `REQUIRE_APPROVAL` | `SIDE_EFFECT_UNKNOWN` |
| 7 | Capability is sensitive and side effects are `NONE` | `REQUIRE_APPROVAL` | `SENSITIVE_CAPABILITY` |
| 8 | No prior rule matches | `DENY` | `UNSUPPORTED_POLICY_CASE` |

Sensitive capabilities are exactly:

- `FILESYSTEM_WRITE`
- `PROCESS_EXECUTE`
- `NETWORK_ACCESS`
- `MCP_INVOKE`

This positive allowlist is intentional. A future capability does not auto-admit merely because it is marked `SideEffect.NONE`.

## Capability, side-effect, and risk semantics

| Fact | Policy meaning |
|---|---|
| Capability not granted | Deny before considering approval. Approval cannot create a capability. |
| `FILESYSTEM_READ` + `NONE` | Allow only when granted and workspace-bound. |
| `MUTATING` | Require approval regardless of `RiskLevel`. |
| `UNKNOWN` | Require approval regardless of `RiskLevel`. Unknown never means safe. |
| Sensitive capability + `NONE` | Require approval. Sensitive capabilities never auto-admit, even when marked `SideEffect.NONE`. |
| `RiskLevel` | Audit and presentation metadata only. It cannot weaken, bypass, or reorder policy rules. |

Examples:

- `RiskLevel.LOW` cannot auto-admit a mutation.
- `RiskLevel.HIGH` does not change the exact granted, workspace-bound, side-effect-free filesystem-read rule.
- A declaration-only Bash spec classifies as approval-required but remains non-executable because executor and handler availability are outside this engine.

## Fail-closed behavior

- Unvalidated inputs return `DENY/INVALID_CALL`; they do not raise into orchestration.
- Missing capability and workspace facts return typed denial.
- Unsupported combinations return `DENY/UNSUPPORTED_POLICY_CASE`.
- The engine never catches or translates handler failures because it never accesses a handler.
- Decisions expose no arguments, secret values, provider payloads, or validation diagnostics.
- Prompt content is not an input and therefore cannot alter a decision.

## Data flow and ownership

```text
provider payload
    -> provider normalization (later work unit)
    -> ToolRegistry.validate_call()
    -> ValidatedToolCall | ToolError
    -> PolicyEngine.decide()
    -> PolicyDecision
    -> approval authority (later work unit, only when required)
    -> typed executor (later work unit, only after admission)
```

Boundary ownership:

| Component | Owns | Must not own in this work unit |
|---|---|---|
| `ToolRegistry` | Schema validation and immutable `ValidatedToolCall` creation | Policy outcomes or execution |
| `PolicyEngine` | Capability/workspace/side-effect classification | Approval state, path containment, handlers, execution |
| Approval layer | Future exact-action plan and one-shot authority | Policy rule overrides |
| Executor | Future handler availability, final plan verification, bounded invocation | Policy or UI decisions |
| `WorkspaceGuard` | Descriptor-relative containment during filesystem access | General tool-policy classification |

The caller must later persist the validated call, registry identity, trusted facts, and policy decision together. Persistence is not part of this slice.

## Test matrix

Add `tests/unit/test_policy_engine.py` and use handlers that fail if invoked.

| Case | Expected result |
|---|---|
| Raw `ToolCall`, dictionary, or arbitrary object | `DENY/INVALID_CALL`; no handler invocation |
| Capability absent, `NONE` | `DENY/CAPABILITY_NOT_GRANTED` |
| Capability absent, `MUTATING` or `UNKNOWN` | Capability denial wins; approval is not offered |
| Workspace missing for an otherwise valid call | `DENY/WORKSPACE_UNAVAILABLE` |
| Granted `FILESYSTEM_READ` + workspace + `NONE` | `ALLOW/READ_ONLY` |
| Granted `FILESYSTEM_READ` + `MUTATING` | `REQUIRE_APPROVAL/SIDE_EFFECTING` |
| Granted `FILESYSTEM_READ` + `UNKNOWN` | `REQUIRE_APPROVAL/SIDE_EFFECT_UNKNOWN` |
| Each sensitive capability + `NONE` | `REQUIRE_APPROVAL/SENSITIVE_CAPABILITY` |
| Any granted capability + `MUTATING` | `REQUIRE_APPROVAL/SIDE_EFFECTING` |
| Any granted capability + `UNKNOWN` | `REQUIRE_APPROVAL/SIDE_EFFECT_UNKNOWN` |
| Same policy input evaluated repeatedly | Equal decisions; no state change |
| Same policy facts with low/high risk variants | Risk never weakens the applicable rule |
| Decision serialization/inspection | Stable strings and correlation; no arguments or free-form diagnostics |
| Domain mutation attempts | Frozen values reject mutation; source capability collection cannot mutate facts |

Prefer parametrized capability/side-effect/risk matrices to keep the authored change below the review threshold.

## File scope

Expected implementation paths:

- `shadow_code/domain/policy.py`
- `shadow_code/policy/engine.py`
- `shadow_code/policy/__init__.py`
- `tests/unit/test_policy_engine.py`

No other production, test, configuration, prompt, provider, UI, persistence, or documentation path belongs in the implementation work unit unless a verified compile/import requirement proves it necessary.

## Evidence and runtime boundary

Future implementation evidence must record:

- focused command: `.venv/bin/python -m pytest -p no:cacheprovider tests/unit/test_policy_engine.py -q`;
- scoped Ruff format check and lint check;
- scoped mypy for the three source modules;
- exact focused results and authored changed-line count.

Runtime scenario: **N/A**. The engine is deliberately pure and has no runtime I/O, provider, UI, filesystem-open, approval, or executor boundary. A deterministic in-process policy matrix is unit evidence, not an execution harness.

## Rollback boundary

Remove `shadow_code/policy/engine.py` and `tests/unit/test_policy_engine.py`, then revert only the new policy domain types and package exports. The committed `WorkspaceGuard`, typed tool contracts, registry, catalog, projections, and legacy production behavior remain intact.

## Review workload forecast

Forecast: **220-320 authored additions plus deletions across four files**. This is below the 400-line review threshold. If implementation exceeds 400 authored changed lines, stop and remove abstractions or split only at a genuine rollback boundary; do not absorb approval or executor work.

## Deferred and base-only defects

Do not repair these in WU-01c.2:

- legacy direct dispatch in `shadow_code/main.py` and `shadow_code/tools/__init__.py`;
- static production Ollama schemas and prompt tool documentation;
- legacy `read_file` absolute-path behavior and handler adapter;
- provider call-ID normalization, approval tokens, executor wiring, persistence, budgets, UI, or prompt activation;
- base-only skill tests:
  - `tests/test_skills.py::TestSkillRegistry::test_register_custom_skill`
  - `tests/test_skills.py::TestSkillRegistry::test_register_overwrites`
  - `tests/test_skills.py::TestBuiltinSkillContent::test_review_skill_mentions_security`
- repository-wide baseline findings: 25 Ruff errors, 10 files needing format normalization, `shadow_code/repl.py:198` mypy `var-annotated`, and low-severity Bandit B101 at `shadow_code/tools/get_language_rules.py:75`;
- `make verify` as read-only evidence, because its coverage target writes `.coverage`.

These are follow-ups or later WU-01c slices, not reasons to expand this rollback boundary.

## Acceptance checklist

- [ ] Policy facts, decisions, dispositions, and reasons are immutable and stable.
- [ ] Only a granted, workspace-bound, `FILESYSTEM_READ` + `SideEffect.NONE` call auto-admits.
- [ ] Unvalidated inputs return typed denial.
- [ ] Missing capability denial takes precedence over approval classification.
- [ ] Missing workspace identity denies.
- [ ] `MUTATING` and `UNKNOWN` always require approval after authority checks.
- [ ] Sensitive capabilities never auto-admit, even with `SideEffect.NONE`.
- [ ] `RiskLevel` remains audit metadata and cannot weaken policy.
- [ ] Unsupported policy combinations deny.
- [ ] No handler is invoked and no side effect occurs in policy tests.
- [ ] No approval/token, executor, filesystem, provider, prompt, UI, or persistence behavior is introduced.
- [ ] Focused tests and scoped check-only quality gates pass with exact evidence recorded.
- [ ] Authored changed lines remain below 400, or implementation stops for a real rollback-boundary split.
- [ ] Rollback removes only this policy-classification unit.
