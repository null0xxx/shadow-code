"""Action-plan digest determinism and one-shot approval authority semantics."""

import re

from shadow_code.domain.approval import (
    ActionPlan,
    ApprovalAuthority,
    ApprovalToken,
    build_action_plan,
    render_action_preview,
)
from shadow_code.domain.policy import WorkspaceIdentity
from shadow_code.domain.tools import ToolCall, ValidatedToolCall
from shadow_code.tools.catalog import BASH_SPEC, READ_FILE_SPEC
from shadow_code.tools.registry import ToolRegistry

WORKSPACE = WorkspaceIdentity(device=100, inode=200)
REGISTRY_DIGEST = ToolRegistry((BASH_SPEC, READ_FILE_SPEC)).digest


def _validated(call_id: str = "call-1", command: str = "id", spec=BASH_SPEC) -> ValidatedToolCall:
    arguments = {"command": command} if spec is BASH_SPEC else {"file_path": "a.txt"}
    call = ToolCall(call_id=call_id, name=spec.name, arguments=arguments)
    return ValidatedToolCall(call=call, spec=spec)


def _plan(
    validated: ValidatedToolCall | None = None,
    preview: str = "preview",
    registry_digest: str = REGISTRY_DIGEST,
    workspace: WorkspaceIdentity = WORKSPACE,
) -> ActionPlan:
    return build_action_plan(
        validated or _validated(),
        registry_digest=registry_digest,
        workspace=workspace,
        preview=preview,
    )


def _mutated(plan: ActionPlan, **changes: object) -> ActionPlan:
    return ActionPlan(**{**plan.model_dump(), **changes})


def test_digest_is_deterministic_sha256_hex() -> None:
    first, second = _plan(), _plan()

    assert first.digest() == second.digest()
    assert re.fullmatch(r"[0-9a-f]{64}", first.digest())


def test_changed_arguments_change_digest_and_preview() -> None:
    base = _plan(_validated(command="id"))
    changed = _plan(_validated(command="whoami"))

    assert base.digest() != changed.digest()
    assert base.canonical_arguments_json != changed.canonical_arguments_json
    assert render_action_preview(_validated(command="id")) != render_action_preview(
        _validated(command="whoami")
    )


def test_changed_preview_changes_digest() -> None:
    assert _plan(preview="one").digest() != _plan(preview="two").digest()


def test_changed_workspace_identity_changes_digest() -> None:
    other = WorkspaceIdentity(device=WORKSPACE.device, inode=WORKSPACE.inode + 1)

    assert _plan().digest() != _plan(workspace=other).digest()


def test_changed_tool_version_changes_digest() -> None:
    plan = _plan()

    assert plan.digest() != _mutated(plan, tool_version="2").digest()


def test_changed_capability_changes_digest() -> None:
    plan = _plan()

    assert plan.digest() != _mutated(plan, capability="network.access").digest()


def test_changed_registry_digest_changes_digest() -> None:
    other_registry = ToolRegistry((READ_FILE_SPEC,))

    assert _plan().digest() != _plan(registry_digest=other_registry.digest).digest()


def test_issue_then_consume_succeeds_exactly_once() -> None:
    authority = ApprovalAuthority()
    plan = _plan()
    token = authority.issue(plan)

    assert token.plan_digest == plan.digest()
    assert authority.consume(token, plan) is True
    assert authority.consume(token, plan) is False  # replay fails


def test_consume_with_changed_plan_fails() -> None:
    authority = ApprovalAuthority()
    token = authority.issue(_plan(preview="one"))

    assert authority.consume(token, _plan(preview="two")) is False


def test_mismatched_digest_burns_the_token() -> None:
    authority = ApprovalAuthority()
    plan = _plan()
    token = authority.issue(plan)

    assert authority.consume(token, _plan(preview="other")) is False
    assert authority.consume(token, plan) is False  # burned; no retry possible


def test_unknown_token_is_rejected() -> None:
    authority = ApprovalAuthority()
    plan = _plan()

    forged = ApprovalToken(token_id="0" * 32, plan_digest=plan.digest())

    assert authority.consume(forged, plan) is False


def test_authority_issues_independent_tokens_per_plan() -> None:
    authority = ApprovalAuthority()
    first_plan, second_plan = _plan(preview="one"), _plan(preview="two")
    first = authority.issue(first_plan)
    second = authority.issue(second_plan)

    assert first.token_id != second.token_id
    assert authority.consume(second, second_plan) is True
    assert authority.consume(first, first_plan) is True
