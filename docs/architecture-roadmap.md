# Shadow Code: trusted local coding agent roadmap

Shadow Code will become a **personal, Python- and Ollama-first coding agent** that is safe enough to use on the owner's workstation, reliable enough to execute multi-step tool workflows, and pleasant enough to keep open throughout a coding session. We will preserve the useful parts of the current project, adapt the interaction patterns that make Easy LLM CLI comfortable, and replace the execution core rather than hiding its weaknesses behind a larger system prompt.

The target outcome is explicit:

- one validated tool protocol from model response to execution;
- code-enforced workspace, capability, approval, and budget boundaries;
- a fully inspectable and editable system prompt whose contents cannot bypass those boundaries;
- an append-only SQLite event history that can reconstruct every completed turn;
- transactional context compression that never separates a tool request from its decision and result;
- one persistent `prompt_toolkit` application, with Rich used as a renderer rather than a second terminal owner;
- measured behavior on the owner's Gemma 4 and Granite models before MCP is introduced.

This document is the implementation contract. A phase is complete only when its acceptance criteria and evidence requirements are met. Planned commands and scenarios are **not** test results.

## Review path

Review the roadmap in this order:

1. Confirm the trust boundary in [Architecture decisions](#architecture-decisions).
2. Verify the code evidence in [Current state](#current-state-verified-by-code-inspection).
3. Challenge the contracts for `PromptCompiler`, `PolicyEngine`, `ToolSpec`, `AgentEngine`, and the event store.
4. Review the first implementation slice in [WU-01](#wu-01--execution-admission-gate-first-implementation-slice).
5. Confirm every later work unit has a dependency, acceptance proof, runtime scenario, and rollback boundary.

## Product boundary

| Question | Decision |
|---|---|
| Primary user | One owner on one Linux workstation. |
| Primary job | Inspect, edit, test, and reason about local code through Ollama-hosted models. |
| Distribution | Personal installation; no public packaging, hosted service, telemetry, Windows, or macOS commitment. |
| Model priority | The owner's Gemma 4 model first, Granite second; other providers are not initial scope. |
| Trust model | Model output is untrusted data. A user-edited prompt is also untrusted with respect to execution authority. |
| UX reference | Easy LLM CLI is an interaction reference, not a codebase, runtime, provider abstraction, or visual identity to copy. |
| Success | The agent selects and calls tools accurately, stops predictably, explains consequential actions, preserves recoverable history, and never gains authority merely because text in the prompt asks for it. |

## Current state, verified by code inspection

The following findings were verified against the extracted working tree on 2026-07-31. The repository currently has no commit on `main`; all files are untracked. This review did not execute the test suite, linter, type checker, security scanner, or live model harness.

### Critical execution risks

| Evidence | Risk | Required correction |
|---|---|---|
| `shadow_code/parser.py:127-147` converts any non-JSON `bash` or `sh` fenced block into a `bash` tool call when no other call was parsed. `shadow_code/main.py:495-527` then executes parsed legacy calls. | An explanatory shell example can become an action. Intent is inferred from Markdown presentation rather than a typed tool request. | Delete raw-shell-fence execution. Ordinary code fences must remain assistant text. Only provider-normalized, schema-valid tool calls can enter policy evaluation. |
| `shadow_code/tools/read_file.py:22-27`, `write_file.py:40-83`, and `edit_file.py:52-109` use `abspath`; only reads consult a small blocked-path list. There is no immutable workspace root or descriptor-relative containment check. | Absolute paths, `..`, symlink pivots, and path swaps can reach files outside the launched project. Writes have weaker checks than reads. | Introduce one Linux `WorkspaceGuard` used by every file tool and by policy before execution. Pin the workspace directory FD; resolve beneath it through an audited `openat2(2)`/`*at` boundary with no-follow flags, and fail closed when required kernel semantics are unavailable. |
| `shadow_code/tools/bash.py:31-40` runs arbitrary strings with `shell=True`, inherits the host environment, and uses the session CWD. Regex warnings in `safety.py` cover a small destructive-pattern list. | A shell is host-level authority. A regex cannot contain shell composition, redirects, substitutions, interpreters, or access outside the workspace. | Treat shell as a separate capability. Preview and approve commands, minimize the environment, enforce budgets, surface sandbox status, and never describe path checks as a shell sandbox. |
| `write_file.py:45-56` permits append without the read-before-write check; `write_file.py:82-83` and `edit_file.py:108-110` write directly to the target. | Partial writes and unreviewed append operations can corrupt files. | Use preview plus read digest, cooperative transaction locking, immediate pre-commit identity/content checks, and descriptor-relative temporary replacement beneath a verified parent FD. Append becomes an explicit mutation mode subject to the same policy. |
| `ToolContext.read_files` records only absolute path strings (`tool_context.py:11-21`). | “Read before write” does not prove the approved bytes are still current. | Record a content digest and stat identity with each read/preview, then reject mutation when current bytes do not match. |

### Protocol and orchestration risks

| Evidence | Risk | Required correction |
|---|---|---|
| Tool definitions exist in `shadow_code/prompt.py`, `ollama_client.py:12-187`, runtime tool classes, and `_TOOL_OUTPUT_LIMITS` in `tools/__init__.py`. | Documentation, provider schemas, validation, execution behavior, and output budgets can drift independently. | Define each tool once as a typed `ToolSpec`; derive provider schema and prompt documentation from the registry. |
| `ollama_client.py:230-235` enables native tools only with `SHADOW_NATIVE_TOOLS=1`, while the static prompt mandates Markdown `tool_call` fences. `main.py` implements both paths. | Two competing protocols produce different history, validation, display, and failure behavior. | Normalize provider-native calls into one domain type. Remove executable Markdown fallback from the default path. |
| The integrated loop, slash commands, persistence, streaming, approval prompts, execution, rendering, and compaction live in `main.py` (647 lines). | State transitions are implicit, difficult to replay, and difficult to test without terminal and model side effects. | Move orchestration into a bounded `AgentEngine`; keep `main.py` as composition and process lifecycle only. |
| Native tool results are formatted without a call ID (`ollama_client.py:266-268`, `conversation.py:42-54`). Missing provider IDs are not normalized. | Multiple calls to the same tool cannot be correlated reliably across streaming, persistence, resume, and UI. | Require an internal `call_id`; preserve provider IDs or generate stable turn-scoped IDs when absent. |
| `MAX_TOOL_TURNS` and consecutive errors bound part of the loop, but there is no duplicate-call detector, turn wall-clock budget, total output budget, or typed retry classification. | A model can repeat an identical call, consume excessive time/output, or retry permanent failures. | Add a `RunBudget`, normalized call signature tracking, terminal stop reasons, and error classes that distinguish validation, policy, cancellation, transient provider, timeout, and execution failures. |

### Persistence and context risks

| Evidence | Risk | Required correction |
|---|---|---|
| SQLite stores only session rows and text messages (`db.py:56-79`). Native tool calls are saved as `[tool calls: N]` in `main.py:424-425`; approval decisions and structured results are not stored. | A resumed session cannot reconstruct what was proposed, approved, executed, or returned. | Replace message-only authority with an append-only event log and rebuild messages/transcript as projections. |
| `Conversation.clear_old_tool_results()` replaces old native `tool` messages with `user` stubs (`conversation.py:70-80`). Compaction and emergency truncation retain the last 20 messages by count (`conversation.py:81-107`). | Role semantics are changed and a tool call, approval, or result can be separated from its causal group. | Compact only complete event ranges at turn/tool-batch boundaries; never delete source events. |
| The system prompt is a static module constant (`prompt.py:1-7`), and project context is injected once into the first user message (`main.py:335-365`). | The owner cannot inspect layered prompt provenance, reload edits predictably, or identify which prompt produced a turn. | Add a deterministic `PromptCompiler`, editable files, `/prompt` commands, and a persisted digest per turn. |

### Terminal UX risks and assets

| Evidence | Assessment | Direction |
|---|---|---|
| `repl.py` creates a new `prompt_toolkit.Application` for each input request; Rich writes outside that application. | Useful input behavior exists, but no component owns the full terminal lifecycle. This invites tearing and limits live state. | One long-lived application owns input, transcript, streaming region, approval focus, and footer. |
| The current UI already has semantic colors, Unicode/ASCII symbols, compact tool output, diffs, context status, and actionable error suggestions (`ui.py`, `theme.py`). | These are reusable concepts. | Preserve semantic roles and fallbacks, not the current Claude-derived branding or renderer coupling. |
| Edit diffs are rendered only after a successful tool execution (`main.py:455-463`). Destructive shell approval uses blocking `input()` (`main.py:433-448`). | The user sees important evidence too late, and approval is disconnected from the main input system. | Render mutation/command previews before execution in a focused inline approval panel. |

## Architecture decisions

These decisions are intentionally stricter than a prompt-only agent. The model proposes; Python code admits, executes, records, and renders.

### Decision 1: keep Python and make Ollama a first-class provider

- Keep Python 3.10+ and the `shadow_code` package.
- Implement an asynchronous `Provider` protocol and an `OllamaProvider` using `httpx.AsyncClient.stream`; replace blocking `requests` in the agent path.
- Keep Ollama request/response details inside the provider adapter. Domain objects do not expose Ollama dictionaries.
- Native tool calls are the initial production protocol for compatible models.
- A future compatibility adapter may parse an explicitly configured structured envelope, but it must produce the same `ToolCall` domain object and pass the same validation and policy gates. Markdown code fences are never executable.

**Tradeoff:** an async HTTP dependency adds migration work, but it prevents network streaming from blocking the terminal event loop and makes cancellation testable.

### Decision 2: model output never carries authority

`PolicyEngine` is code, not prompt prose. It receives immutable session capabilities, a normalized `ToolCall`, `ToolSpec` metadata, and current workspace/execution facts. It returns a typed decision:

```text
allow | require_approval | deny
```

The initial capability vocabulary is:

```text
filesystem.read
filesystem.write
process.execute
network.access
mcp.invoke
```

Rules:

- the workspace root is fixed at session creation (`--workspace` defaults to launch CWD);
- the model cannot change the workspace root, capabilities, approval mode, or sandbox mode;
- changing workspace is a user command that starts a new authority boundary;
- file reads and writes must pass `WorkspaceGuard` before handler invocation;
- a read-only, side-effect-free tool may be admitted automatically; every tool declared side-effecting or side-effect-unknown requires a fresh explicit approval for every call;
- approval is a one-shot token bound to an immutable `ActionPlan`: call ID/schema version, `ToolSpec` name/version, canonical validated arguments, capability, workspace identity and CWD, preview digest, and applicable executable/environment/network/MCP/sandbox facts;
- the executor recomputes the canonical action-plan digest immediately before execution; any changed fact, stale preview, reused token, or mismatched digest rejects the call without a side effect;
- remembered grants and policy-wide auto-accept are out of scope; adding either requires a separate product decision and threat-model review;
- shell is not declared “sandboxed” merely because CWD is inside the workspace;
- if a supported Linux sandbox is unavailable, the UI says so; strict mode denies shell, while explicit permissive mode requires approval for each unclassified command;
- prompt text can recommend behavior but cannot alter a policy result.

### Decision 3: the system prompt is layered, visible, and editable

`PromptCompiler` assembles a deterministic prompt snapshot for each turn from these ordered layers:

1. versioned built-in operational defaults;
2. user file: `~/.config/shadow-code/prompts/default.md`;
3. project file: `<workspace>/.shadow-code/instructions.md`;
4. optional session overlay held in session state;
5. generated tool documentation from the active `ToolSpec` registry;
6. generated runtime facts with a strict size and secret-redaction policy.

Later layers override behavioral guidance by explicit section key; generated tool schemas and runtime facts remain generated so they cannot drift from execution. The compiled prompt is fully viewable. There is no hidden “safety prompt”; safety resides in `PolicyEngine` and executors.

Commands:

```text
/prompt show
/prompt edit
/prompt reload
/prompt diff
/prompt validate
/prompt sources
/prompt history
/prompt rollback <snapshot>
```

Compilation rules:

- normalize line endings to UTF-8 LF;
- reject unreadable files and over-limit layers with a typed error rather than silently dropping them;
- snapshot at turn start; a file edit affects the next turn, never a turn already in flight;
- persist an immutable snapshot containing the exact compiled bytes and normalized bytes of every source layer, plus source kind/path, precedence, source and compiled SHA-256 digests, compiler version, tool-registry digest, runtime-facts digest, timestamps, and parent/rollback provenance;
- treat digests as integrity checks, not snapshot contents: `/prompt history` lists restorable snapshots, and `/prompt rollback <snapshot>` stages the stored bytes, verifies every digest and compatibility invariant, validates the prompt, then atomically activates the selected snapshot and records the previous/new identities;
- rollback failure leaves both active prompt and editable source files unchanged; rollback never silently substitutes current file contents for missing snapshot bytes;
- never include arbitrary environment variables, tokens, or secret file contents in runtime facts.

### Decision 4: `ToolSpec` is the single source of tool truth

Use Pydantic v2 models for argument and result boundaries. A tool is registered through one immutable specification:

```python
ToolSpec(
    name,
    version,
    description,
    args_model,
    handler,
    capability,
    risk,
    side_effects,
    timeout_seconds,
    max_output_chars,
    idempotency,
    parallel_safety,
    renderer_hint,
)
```

From that object the system derives:

- Ollama function schema;
- runtime argument validation;
- prompt tool documentation;
- policy metadata;
- output limits;
- UI labels and preview renderer selection;
- a stable registry digest.

Unknown fields are rejected by default. Validation failures are structured and returned to the model at most once per unique invalid call signature. Tool handlers receive validated argument models and an `ExecutionContext`; they do not receive raw dictionaries or global registry state.

**Tradeoff:** Pydantic adds a dependency, but eliminates hand-maintained schema/runtime duplication and produces precise validation errors for small local models.

### Decision 5: file mutation has an explicit, limited concurrency guarantee

`MutationPlan` contains the pinned workspace, parent, and existing-target object identities; normalized relative target; operation; before digest/stat; proposed bytes; diff; byte count; and risk metadata. As the file payload of the approved `ActionPlan`, it authorizes only those exact identities, bytes, and execution facts. Approval does **not** prove that a pathname will remain beneath the workspace, or provide atomic existing-target content/path compare-and-swap, against a hostile concurrent namespace mutator.

Direct in-place apply is designed for model-controlled path/symlink input and ordinary local editing. Shadow Code processes cooperate through one exclusive per-workspace transaction lock. A concurrent editor that honors the lock is serialized; a normal editor that does not honor it is protected only by immediately-before-commit stale-state checks. A malicious or deliberately racing, uncooperative same-UID process that can rename or swap workspace ancestry is outside this direct-apply guarantee. The lock coordinates cooperating writers; it is not a security boundary.

Direct write algorithm:

1. acquire the cooperative workspace transaction lock before the final preview snapshot and hold it through commit, directory fsync, readback, and event recording;
2. open and pin the workspace directory FD, then resolve the target parent relative to it with `openat2(2)` using `RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS | RESOLVE_NO_MAGICLINKS | RESOLVE_NO_XDEV`; reject absolute paths, `..`, magic links, mount escapes, or unsupported semantics;
3. pin the parent FD and identity; open an existing target relative to it with no-follow semantics and capture bytes, digest, object identity, mode, and absence/presence state;
4. create the `MutationPlan`, display it, and obtain fresh approval for the complete immutable `ActionPlan`;
5. create the temporary file through the pinned parent FD using exclusive, no-follow descriptor-relative operations; write, flush, fsync, and apply the intended mode;
6. immediately before commit, re-resolve the planned parent beneath the pinned workspace FD and require the planned parent identity; inspect the target through that verified parent and require its planned identity, content digest, mode, and presence state; on any observed change, abort and remove the temporary file;
7. replace/create through that parent FD with `renameat2(2)`/equivalent semantics, fsync the directory, read back through the parent FD, and record the resulting identity and digest.

Descriptor-relative resolution removes path-string and model-selected symlink fallbacks, while replacement remains atomic only as a replacement operation. A verified directory FD follows its object if another process renames it, and Linux offers no general atomic “replace this pathname only if its prior inode and digest still match” primitive. Tests therefore prove rejection of changes observed at defined checks, not impossible syscall-atomic pathname/content CAS.

No append or path-based fallback exists. Exact-text edit remains a planner that emits a `MutationPlan`. If required kernel/filesystem primitives or the stable-workspace preconditions cannot be established, direct apply is disabled. Strict/high-assurance mode also disables direct apply unless the runtime has separately proven OS-enforced non-movable ancestry; the initial implementation treats that capability as unavailable. In either case Shadow Code writes a reviewed patch/artifact in isolated staging for manual or out-of-band application, records `exported` rather than `executed`, and never reports that the live workspace was mutated.

### Decision 6: `AgentEngine` is a bounded state machine

The engine owns state transitions; the UI only projects emitted events.

```text
IDLE
  -> PREPARING
  -> STREAMING
  -> COLLECTING_CALLS
  -> VALIDATING_CALLS
  -> EVALUATING_POLICY
  -> AWAITING_APPROVAL (zero or more calls)
  -> EXECUTING
  -> APPENDING_RESULTS
  -> PREPARING_NEXT_STEP | COMPLETED

Any active state may end as CANCELLED, FAILED, or BUDGET_EXHAUSTED.
```

Invariants:

- one `run_id`, one `turn_id`, and one `call_id` identify every causal chain;
- missing provider call IDs are replaced by stable `turn_id/sequence` IDs;
- calls are collected before execution; partial streamed arguments never execute;
- calls execute sequentially initially; parallel execution is added only for specs marked parallel-safe and independent by policy;
- each call has exactly one terminal result: success, denied, cancelled, validation error, timeout, or execution error;
- a denied or cancelled call is returned to the model as a terminal result and is never retried automatically;
- the same normalized call signature may execute once by default; repetition requires a changed causal input and remaining budget;
- cancellation terminates provider streaming or the process group and records cleanup evidence.

Initial `RunBudget` fields:

```text
max_model_steps
max_tool_calls
max_calls_per_step
turn_deadline
per_tool_deadline
max_total_tool_output_chars
max_consecutive_failures
max_duplicate_signature_count
```

Retry is limited to typed transient provider failures and occurs before a tool call is admitted. Validation, policy, denial, cancellation, and deterministic tool failures are not transient.

### Decision 7: SQLite events are authoritative

SQLite remains because it is local, inspectable, transactional, and already present. The new schema uses WAL, foreign keys, explicit migrations, and a single writer service.

Core tables:

```text
sessions
runs
events(sequence, session_id, run_id, turn_id, event_type, schema_version,
       occurred_at, payload_json)
prompt_snapshots
context_snapshots
schema_migrations
```

Authoritative event types include:

```text
user_message_recorded
turn_started
prompt_compiled
prompt_snapshot_activated
prompt_snapshot_rolled_back
provider_response_completed
tool_call_proposed
tool_call_validated
policy_decided
approval_recorded
tool_execution_started
tool_execution_finished
turn_completed
turn_failed
turn_cancelled
context_snapshot_created
```

Transient token deltas may drive UI updates but are not individually persisted. The completed assistant payload, structured calls, results, timings, usage, prompt snapshot identity, policy decision, action-plan digest, approval token consumption, and stop reason are persisted. Transcript and conversation messages are disposable projections rebuilt from events.

Event append and projection checkpoint update occur in one transaction. An idempotency key prevents duplicate terminal events after interruption. Migrations never destroy the legacy database; import is explicit and reversible.

### Decision 8: context compression is transactional and causal

`ContextBuilder` selects only completed event ranges. A compressible range must end after a completed turn and may not contain a call without its policy decision and terminal result.

Compression protocol:

1. select a closed sequence interval;
2. build a deterministic source projection and its digest;
3. ask the configured summarizer for a typed snapshot containing goals, decisions, constraints, changed files, verification evidence, failures, and next actions;
4. validate required fields and referenced paths against source events;
5. append `context_snapshot_created` with covered range and source digest in one transaction;
6. build future prompts from the snapshot plus later events;
7. retain original events permanently unless the user explicitly deletes the session.

If summarization or validation fails, no snapshot becomes active. Emergency reduction drops only oldest complete, already-snapshotted turn groups from the provider projection; it never truncates an arbitrary message count and never mutates the event log.

### Decision 9: one terminal owner, two rendering responsibilities

`prompt_toolkit.Application` owns the terminal, keyboard focus, async event loop, and layout for the lifetime of the process. Rich renders Markdown, syntax-highlighted code, tables, and unified diffs into an in-memory console; the resulting ANSI is converted to `prompt_toolkit` formatted text. Rich never writes directly while the application is active.

Layout:

```text
completed transcript (scrollable, cached)
live assistant / tool group region
focused approval or error region
multiline editor
single-line footer
```

Required interaction states:

```text
validating -> awaiting approval -> running -> success | failed | denied | cancelled
```

Visual strategy:

- terminal-native background rather than a forced dark or light canvas;
- restrained semantic color roles for focus, tool, success, warning, error, diff add/remove, and muted text;
- Shadow Code identity instead of Claude orange or Easy LLM CLI branding;
- one consistent component vocabulary, compact density, and no decorative motion;
- Unicode with ASCII fallback, ANSI theme, and `NO_COLOR` support;
- width-aware 40/80/120-column layouts, height-aware truncation, and explicit expand/collapse;
- WCAG-equivalent contrast checks for committed themes; color is never the only state signal.

### Decision 10: MCP is an adapter after the core is proven

MCP is not a shortcut around `ToolSpec`, policy, budgets, or events. Each server owns its client and lifecycle; there is no process-global client.

- names: `mcp.<server>.<tool>`;
- supported transports are introduced independently;
- environment and headers are explicit allowlists and are redacted from logs;
- discovered schemas are sanitized and wrapped in `ToolSpec`;
- server, tool, capability, timeout, output limit, and trust decision are persisted;
- no server or tool is trusted by default;
- network MCP requires `network.access`; all MCP calls require `mcp.invoke`;
- disconnect, cancellation, and process cleanup are terminal events.

## Easy LLM CLI: adapt versus reject

The reference was inspected at commit `e2ff5bcc5cb0334830dabc6ba0fc5b2ec74b4d7b`.

| Adapt the pattern | Why it helps Shadow Code |
|---|---|
| Persistent application with completed history separated from live state | Prevents output tearing and keeps the active operation understandable. |
| Grouped tool calls with explicit lifecycle status | Makes multi-call model turns reviewable without flooding the transcript. |
| Inline confirmation with command or diff preview and focused choices | Keeps consequential decisions in context and avoids raw `y/n` reads. |
| Terminal-size-aware truncation and “show more” behavior | Preserves control on small terminals without losing access to full evidence. |
| Footer with model, CWD/workspace, Git branch, context, and permission state | Makes invisible runtime authority visible. |
| Semantic theme roles, ANSI/no-color variants, Markdown, code, and diff rendering | Produces a comfortable product UI without coupling behavior to colors. |
| Separate scheduler/core and UI projections | Lets deterministic tests drive the engine without rendering a terminal. |
| Tool registry and typed confirmation details | Supports one path from declaration to preview and policy. |

| Reject the implementation | Reason |
|---|---|
| React/Ink and the Node monorepo | It adds a second ecosystem without improving the Python/Ollama core. |
| Gemini CLI domain types as the internal protocol | Provider-specific types would leak into every engine, tool, and persistence boundary. |
| Converting custom providers into Gemini-shaped content | Shadow Code should normalize at its own provider boundary, not inherit Gemini semantics. |
| A global MCP client or trust shortcut | Server lifecycles and authority must remain isolated and auditable. |
| Copying colors, ASCII art, labels, or branding | Interaction quality is the reference; Shadow Code needs its own identity. |
| Auto-accept as an early convenience feature | It would weaken the boundary before policy and evidence are proven. |
| Prompt text as the primary security control | It is advisory and can be edited or ignored by the model. |

## Target package boundaries

The exact filenames may evolve within a work unit, but dependencies must point inward and each responsibility has one owner.

```text
shadow_code/
  cli.py                    # arguments and process lifecycle
  bootstrap.py              # dependency composition only
  domain/
    messages.py
    tools.py                # ToolSpec, ToolCall, ToolResult
    events.py
    policy.py
    budgets.py
  providers/
    base.py
    ollama.py
  prompts/
    compiler.py
    sources.py
  policy/
    engine.py
    workspace.py
    approvals.py
  tools/
    registry.py
    executor.py
    filesystem.py
    shell.py
    builtins/...
  agent/
    engine.py
    state.py
  persistence/
    event_store.py
    migrations.py
    projections.py
  context/
    builder.py
    compressor.py
  ui/
    application.py
    view_models.py
    renderers.py
    theme.py
  mcp/                     # absent until its work unit
tests/
  unit/
  contract/
  transcript/
  integration/
  runtime/
```

Dependency rule: `domain` imports no adapter. Providers, persistence, tools, and UI implement domain-facing protocols. `bootstrap.py` is the only place that constructs concrete adapters.

## Delivery and evidence rules

Every work unit is a reviewable behavior, not a file-type batch.

- Code, focused tests, migrations/fixtures, and user-facing documentation for a behavior ship together.
- Before functional verification, run source-mutating formatters once; all later checks are check-only.
- Record the exact focused command and exact result in the work-unit evidence. Do not translate a missing tool into a pass.
- Record one runtime scenario and exact result, or state `N/A` with a reason when the unit has no runtime boundary.
- If a work unit forecasts more than 400 authored changed lines, split it along the rollback boundaries below before implementation; generated fixtures remain in snapshot identity.
- A later phase never repairs an unproven dependency silently. Stop and resolve the failed acceptance criterion first.
- Conventional commits only; no AI attribution or `Co-Authored-By` trailers.

## Phased work units

### Phase 0 — freeze the source and make evidence reproducible

#### WU-00 — baseline and local verification harness

**Depends on:** approved roadmap.

**Outcome:** preserve the extracted source as an immutable Git baseline and make the existing verification commands reproducible without changing runtime behavior.

**Scope:**

- initialize/confirm Git metadata and commit the untouched extracted source plus this roadmap as the baseline story;
- create a project-local virtual environment and lock or constrain development dependencies;
- document exact formatter, linter, type-check, security, unit-test, and coverage commands;
- capture existing behavior with transcript fixtures where it can be observed without changing production code.

**Acceptance criteria:**

- `git status` is clean after the baseline commit;
- the baseline commit can be checked out and imports the package in the documented environment;
- every verification command either completes with a recorded result or is reported unavailable/failed with its diagnostic;
- no production behavior is intentionally changed.

**Focused checks:** `python -m pytest tests/test_parser.py tests/test_tools.py tests/test_db.py -q`; then the documented full local verification command.

**Runtime scenario:** launch the baseline CLI against a disposable Ollama test session, send one text-only prompt, then exit. Record model tag and observable result; do not allow mutation tools.

**Rollback boundary:** delete the local environment and revert only baseline-support documentation/configuration; the extracted source bytes remain recoverable from the baseline commit.

**Risk:** existing tests may encode unsafe behavior. A passing baseline records behavior; it does not approve it.

### Phase 1 — establish the execution boundary

#### WU-01 — execution admission gate (first implementation slice)

**Depends on:** WU-00.

**Outcome:** no model-produced action reaches a handler unless one typed specification validates it and code-enforced policy admits it.

**Scope:**

- add domain `ToolCall`, `ToolResult`, `ToolError`, `ToolSpec`, capability, risk, side-effect, and policy-decision types;
- introduce the Pydantic-backed registry and migrate `read_file` plus a non-executing `bash` declaration as proof;
- generate Ollama schema and prompt tool documentation from the registry;
- implement immutable `WorkspaceGuard` and minimal `PolicyEngine`;
- remove raw `bash`/`sh` fence conversion and disable executable Markdown tool fallback;
- route admitted calls through one executor entry point;
- keep legacy tools behind a non-default compatibility boundary until migrated.

**Acceptance criteria:**

- a `bash` code fence is returned as assistant text and emits zero tool-call events;
- unknown tool names, unknown fields, wrong types, and missing required fields never invoke a handler;
- `read_file` opens only normalized relative paths beneath the pinned workspace FD and rejects absolute paths, `..`, symlinks, magic links, and unsupported descriptor-containment semantics;
- provider schema, runtime validation, prompt documentation, and policy metadata originate from the same `ToolSpec` instance;
- editing prompt text cannot change a `deny` into `allow`;
- read-only calls may auto-admit, but each side-effecting/unknown call requires a fresh one-shot approval whose action-plan digest matches at execution; changed arguments, preview, CWD/workspace, tool version, capability, or sandbox/execution facts reject it;
- an approval token cannot authorize a second call, survive a changed action plan, or become a remembered grant;
- one valid in-workspace read still succeeds and returns a typed, bounded result.

**Focused tests:**

```text
tests/unit/test_tool_spec.py
tests/unit/test_workspace_guard.py
tests/unit/test_policy_engine.py
tests/unit/test_action_plan_approval.py
tests/unit/test_parser_non_execution.py
tests/contract/test_registry_schema_parity.py
```

Run the exact new focused subset plus the existing parser/tool tests and record results.

**Runtime scenario:** in a disposable workspace containing a normal file, an outside file, and a symlink to the outside file, feed a fake provider transcript with four read proposals and one explanatory shell fence. Exactly one read may reach the handler; no shell call may execute.

**Rollback boundary:** revert the new domain/policy/registry modules and the single composition change; restore the baseline parser and registry together. Do not retain half of the admission path.

**Risks:** migrating too many tools will inflate the slice; only one real read handler and one non-executing shell declaration prove the contract. Pydantic error formatting must be normalized so model-specific noise does not leak into domain tests.

#### WU-02 — safe file mutation plans

**Depends on:** WU-01.

**Outcome:** file changes are previewable and atomically replaced under the declared stable-workspace/cooperative-writer threat model, with strict mode falling back to patch export.

**Scope:** migrate `write_file` and exact-text `edit_file`; implement read snapshots, `MutationPlan`, exact action-plan approval binding, a cooperative per-workspace transaction lock, the Linux descriptor-relative syscall adapter, immediate pre-commit identity/content validation, atomic replacement, isolated patch export, mode preservation, and post-write readback.

**Acceptance criteria:** append cannot bypass read/preview/policy; duplicate edit matches fail closed; every mutation consumes a new exact-plan approval; cooperating Shadow Code writers serialize on the workspace lock; all opens/temp creation/replacement are descriptor-relative and no-follow; any identity, content, mode, presence, or approval fact changed before the immediate commit check aborts; failure before replacement leaves the original intact; successful direct apply records before/after identities and digests; the lock is never represented as protection from uncooperative processes; strict mode refuses direct apply while enforced non-movable ancestry is unavailable; unsupported primitives or unstable-workspace preconditions produce only an isolated reviewed patch marked `exported`, never an executed mutation.

**Focused tests:** mutation planning, approval replay/mismatch, cooperative-lock serialization, encoding, mode, empty file, Unicode/Georgian text, new-file parent resolution, and a deterministic interleaving harness that injects content, path, symlink, parent, or mount changes immediately before each defined verification boundary and proves observed changes abort; also short write/fsync/rename failure, primitive/precondition failure to patch export, exported-versus-executed event truth, and strict-mode refusal without an enforced non-movable-ancestry capability. These tests do not claim coverage of an uncooperative swap in the final syscall window.

**Runtime scenario:** request and approve a one-line edit in a stable disposable Git repository, then verify the diff, lock lifecycle, immediate pre-commit checks, readback, and approval-consumption events. Inject a parent/symlink swap before the final check and require abort with an unchanged outside sentinel. Repeat in strict mode without enforced ancestry: the live tree remains unchanged, a reviewed patch is exported, and no event or UI label says the mutation executed.

**Rollback boundary:** remove mutation tools and planners while retaining WU-01 read-only admission. The CLI remains useful in read-only mode.

**Risk:** Linux does not provide general atomic content-plus-path CAS against a hostile same-UID namespace mutator. Direct apply deliberately offers narrower cooperative/stable-workspace guarantees; high-assurance use exports a patch for out-of-band application instead of overstating confinement.

#### WU-03 — controlled process execution

**Depends on:** WU-01; uses WU-02 approval UI contract when available.

**Outcome:** shell execution is explicit host authority with predictable timeout, cancellation, environment, output, and approval behavior.

**Scope:** introduce `ProcessExecutor`, command preview, minimal environment policy, process-group cancellation, sandbox capability detection, stdout/stderr byte budgets, and terminal cleanup.

**Acceptance criteria:** shell never runs from a code fence; strict mode denies when required sandboxing is unavailable; permissive mode visibly labels unconfined execution and requires a fresh approval bound to the exact command, arguments, CWD/workspace, environment digest, executable resolution, and sandbox facts; any change or replay rejects; timeout/cancel kills the process group; output truncation records removed byte counts; secrets from the parent environment are absent unless explicitly allowed.

**Focused tests:** approval denial/replay and action-plan mismatch, environment filtering, timeout, child-process cleanup, output truncation, command substitution/redirection classification, cancellation race, invalid encoding.

**Runtime scenario:** execute a harmless command, a long-running child tree cancelled by the user, and a command that attempts to print a filtered sentinel environment variable. Record cleanup evidence.

**Rollback boundary:** unregister `bash` and retain WU-01/WU-02 file capabilities.

**Risk:** no parser can make a shell safe. The UI and documentation must never overstate guarantees when a kernel sandbox is absent.

### Phase 2 — normalize prompts, providers, and history

#### WU-04 — layered `PromptCompiler`

**Depends on:** WU-01 registry.

**Outcome:** the owner can inspect, edit, validate, reload, audit, and roll back behavioral instructions without creating a second source of tool truth.

**Scope:** implement prompt sources, deterministic compiler, immutable snapshots containing compiled/source bytes and provenance, content-addressed integrity checks, redacted runtime facts, atomic active-snapshot switching, file watching/reload semantics, and `/prompt show|edit|reload|diff|validate|sources|history` plus `/prompt rollback <snapshot>`.

**Acceptance criteria:** layer order and override rules are deterministic; identical inputs produce identical bytes/digest; each snapshot can reproduce its exact compiled and normalized source bytes without consulting current files; a source edit affects only the next turn; history exposes provenance; rollback verifies stored bytes/digests and compatibility, validates before activation, switches active identity atomically, emits auditable previous/new snapshot events, and leaves active state unchanged on any failure; generated tool docs match registry digest; unreadable/oversize sources fail visibly; prompt contents cannot grant a capability; every turn identifies the exact prompt snapshot.

**Focused tests:** missing files, Unicode, CRLF normalization, precedence, size limits, exact byte round-trip, corrupt/missing snapshot blob, provenance, history ordering, successful rollback, validation/compatibility rejection, crash during activation, audit events, atomic source replacement, registry change, secret redaction, and in-flight snapshot isolation.

**Runtime scenario:** capture snapshot A, edit/reload to B, run a text-only turn, use history to roll back to A, verify exact compiled/source bytes and rollback audit chain after restart, then corrupt a disposable snapshot and prove failed rollback leaves B active; policy behavior remains unchanged throughout.

**Rollback boundary:** select the built-in prompt source adapter while leaving policy and registry intact; user prompt files remain untouched.

**Risk:** excessive prompt customization can reduce model tool accuracy. `/prompt validate` checks structure and size, not semantic quality; model evaluation remains necessary.

#### WU-05 — provider-neutral Ollama streaming contract

**Depends on:** WU-01 domain types and WU-04 prompt snapshots.

**Outcome:** Ollama text, usage, stop reasons, and native calls become one typed stream independent of UI and engine.

**Scope:** add async provider protocol, `OllamaProvider`, fragmented stream assembly, argument normalization, generated IDs, cancellation, timeouts, and captured provider fixtures with secrets removed.

**Acceptance criteria:** partial arguments never execute; mixed text and calls are preserved; multiple calls maintain order and identity; malformed payloads produce typed provider errors; cancellation closes the response; provider dictionaries do not cross the adapter boundary.

**Focused tests:** recorded NDJSON for text, one call, multiple calls, mixed content, missing IDs, malformed JSON, HTTP error, disconnect, timeout, cancellation, unknown fields.

**Runtime scenario:** text-only and tool-request prompts against the exact installed Gemma 4 and Granite tags; record provider capability observations without changing contracts to fit one transcript.

**Rollback boundary:** keep the fake provider and WU-01 tests; restore the old Ollama client only for text-only diagnostic use, never for tool execution.

**Risk:** Ollama/model versions differ in native-call shape. Normalize observed variants at the provider boundary and pin fixture provenance.

#### WU-06 — append-only event authority

**Depends on:** WU-01 event types; may proceed in parallel with WU-05 after shared types freeze.

**Outcome:** every causal action needed for resume or audit survives process exit.

**Scope:** schema migrations, event append API, prompt snapshots, transaction/idempotency rules, transcript/message projections, explicit legacy session importer, and integrity diagnostics.

**Acceptance criteria:** complete event chains rebuild the same transcript and provider messages; tool calls/results retain IDs and typed payloads; duplicate terminal append is idempotent; a transaction interruption leaves no partial projection checkpoint; legacy DB is never destructively migrated.

**Focused tests:** fresh DB, migration upgrade/downgrade policy, WAL setup, foreign keys, append ordering, concurrent-reader behavior, interruption, duplicate key, corrupt payload, projection rebuild, legacy import copy.

**Runtime scenario:** complete a fake-provider call, terminate the process, resume, and verify the transcript, pending-state detection, prompt digest, policy decision, and result are identical.

**Rollback boundary:** preserve the new database file and disable its adapter; no rollback deletes user history. Re-enable legacy read-only session display if needed.

**Risk:** event schema churn creates permanent compatibility cost. Version every payload from the first migration.

### Phase 3 — make tool use deterministic

#### WU-07 — bounded `AgentEngine`

**Depends on:** WU-01 through WU-06.

**Outcome:** multi-step tool work follows explicit states, budgets, and terminal outcomes with no UI dependency.

**Scope:** implement engine states, event emission, call collection, validation, policy/approval handoff, sequential execution, result append, typed errors, budgets, duplicate detection, and cancellation.

**Acceptance criteria:** every proposed call reaches exactly one terminal result; no handler runs before stream completion, validation, and policy; denial is not retried; duplicate calls stop at configured budget; step/tool/time/output limits produce `BUDGET_EXHAUSTED`; resumed incomplete executions fail closed and ask for a user decision rather than guessing.

**Focused tests:** deterministic transcript cases for no tools, one tool, multi-step sequence, multi-call batch, malformed args, denial, cancellation at every active state, transient provider retry, permanent failure, repeated call, each budget, crash after start/before finish.

**Runtime scenario:** ask each target model to inspect two files, propose one edit, run one focused check, and summarize. Use a disposable repository and require the same event invariants even if model success rates differ.

**Rollback boundary:** keep provider, event store, prompt compiler, and tools accessible through diagnostic commands; remove the autonomous loop as one unit.

**Risk:** small models may need better prompt examples. Do not weaken validation or increase budgets until transcript evidence identifies the failure class.

#### WU-08 — transactional context building and compression

**Depends on:** WU-06 event authority and WU-07 completed-turn semantics.

**Outcome:** long sessions shrink model context without losing causal tool groups or source history.

**Scope:** context projection, closed-range selector, typed summary snapshot, validation, active-snapshot selection, emergency complete-turn reduction, and context diagnostics.

**Acceptance criteria:** no selected range splits a call/decision/result chain; failed summary creates no active snapshot; original events remain queryable; snapshot source digest detects changed projection logic; provider messages after rebuild remain protocol-valid; emergency reduction removes only complete snapshotted groups.

**Focused tests:** boundary selection, pending approvals, denied calls, multi-call batches, summary schema failure, hallucinated file reference, digest mismatch, overlapping snapshots, Unicode, token estimator error, emergency reduction.

**Runtime scenario:** replay a long synthetic session past thresholds, compact, resume, and complete a pending coding goal; compare event history and active provider projection before and after.

**Rollback boundary:** ignore context snapshots and rebuild from original events; lower session length manually rather than deleting data.

**Risk:** summaries are lossy. Preserve verifiable decisions and evidence structurally, and measure task continuation rather than summary prose quality alone.

### Phase 4 — deliver the terminal product

#### WU-09 — persistent terminal shell

**Depends on:** WU-07 engine event stream; can use fake provider initially.

**Outcome:** one `prompt_toolkit` application owns the session without tearing or blocking on input.

**Scope:** persistent layout, async engine bridge, transcript projection, live response area, multiline editor, history/completion, focus model, footer, resize behavior, Unicode/ASCII, ANSI, and `NO_COLOR` themes.

**Acceptance criteria:** Rich performs no direct active-terminal writes; resize at 40/80/120 columns preserves input and focus; Ctrl+C cancels active work but does not corrupt the terminal; model/workspace/Git/context/permission state remains visible; plain/ASCII mode preserves all semantic labels.

**Focused tests:** view-model snapshots, key bindings, resize, narrow/short terminal, focus transitions, cancellation, pasted multiline/Unicode text, no-color output, renderer sanitization.

**Runtime scenario:** run a ten-minute fake-provider session with streaming, resize, scrollback, command completion, cancellation, and resume; inspect terminal cleanup after normal exit and injected failure.

**Rollback boundary:** retain the engine and expose a minimal line-oriented diagnostic client; UI removal does not alter execution or stored history.

**Risk:** full-screen terminal control can fight native scrollback. Prototype the ownership strategy against the owner's terminal before freezing it; preserve a non-full-screen diagnostic mode.

#### WU-10 — tool lifecycle, previews, and approvals

**Depends on:** WU-02/WU-03 preview contracts and WU-09 shell.

**Outcome:** Easy LLM CLI's strongest interaction patterns are adapted into Shadow Code's own compact visual system.

**Scope:** grouped calls, status transitions, diff/command previews, inline single-focus approval, output truncation/expand, failure guidance, Markdown/code/diff renderers, and Shadow Code theme tokens.

**Acceptance criteria:** the user sees every approval-bound action-plan fact and digest before approval; each side-effecting call receives its own fresh, single-focus control and one-shot token; changed/stale plans return to preview rather than inheriting consent; denial/cancel is final and visible; long output cannot push the editor off-screen; state remains understandable without color; rendered ANSI/model text cannot inject terminal control sequences.

**Focused tests:** every lifecycle state, sequential fresh approvals in a batch, approval replay and stale-plan rendering, preview overflow, huge paths/commands, zero-width and Georgian text, malicious ANSI, denial, external mutation/authority-fact change while awaiting approval, exported-versus-executed labels, error suggestions.

**Runtime scenario:** one multi-call turn containing reads, an edit preview, a denied shell command, a successful focused test, and an expanded result; validate at narrow and normal widths.

**Rollback boundary:** fall back to WU-09 plain event rows; the engine continues to enforce all decisions.

**Risk:** visual polish can obscure evidence. Payload, status, capability, and result take priority over decoration.

### Phase 5 — prove model behavior and harden operations

#### WU-11 — Gemma/Granite evaluation matrix

**Depends on:** WU-07 through WU-10.

**Outcome:** tool-calling quality is measured on the exact installed models instead of inferred from prompt quality.

**Scope:** versioned scenario corpus, deterministic fake-provider regression suite, opt-in live harness, scoring, failure taxonomy, prompt/model provenance, and comparison reports.

**Metrics:** correct tool choice, argument validity, path accuracy, calls-to-completion, duplicate rate, denial compliance, recovery from malformed response, budget adherence, edit correctness, verification honesty, latency, and peak context.

**Required scenarios:** read-only orientation, targeted grep/read, exact edit, new file, focused test, multi-step failure recovery, denied command, cancellation, prompt injection in repository content, symlink escape, stale preview/concurrent-mutation detection, strict patch export, context compression continuation, malformed native call, and repeated-call termination.

**Acceptance criteria:** the corpus runs unchanged for Gemma 4 and Granite; every score links to raw redacted events and exact model/prompt/tool-registry digests; release thresholds are declared before tuning; safety invariant failures are release blockers regardless of aggregate score.

**Focused checks:** deterministic corpus tests plus schema checks for live-run reports.

**Runtime scenario:** execute the full live matrix in disposable workspaces only, one model at a time, with a cost/time forecast and captured cleanup evidence.

**Rollback boundary:** live harness remains opt-in and removable; deterministic fixtures continue to protect engine behavior.

**Risk:** tuning to a tiny corpus produces brittle prompts. Keep holdout scenarios and categorize failures before changing prompt, model options, or engine behavior.

#### WU-12 — operational polish

**Depends on:** WU-11 evidence.

**Outcome:** the personal CLI starts, upgrades, diagnoses, and recovers predictably on the owner's system.

**Scope:** configuration diagnostics, database backup/restore, migration preview, Ollama/model capability checks, crash recovery, log redaction, resource cleanup, and a single local install/update command.

**Acceptance criteria:** startup names every active authority boundary and unavailable capability; upgrade is reversible; DB backup restores; crash leaves no orphan child; logs contain no configured sentinel secrets; unsupported model features fail with actionable diagnostics.

**Focused tests:** configuration precedence, corrupt config, migration failure, backup/restore, redaction, stale process metadata, unavailable Ollama, missing model, incompatible tool support.

**Runtime scenario:** install from a clean checkout, run diagnostics, complete a safe edit, simulate interrupted upgrade/session, recover, and uninstall without deleting session data unless explicitly requested.

**Rollback boundary:** restore the previous local executable/config and database backup independently.

**Risk:** personal-only assumptions can become invisible dependencies. Diagnostics must state them explicitly rather than pretending portability.

### Phase 6 — extend through MCP only after the core passes

#### WU-13 — isolated MCP adapter

**Depends on:** WU-11 thresholds passing and WU-12 recovery controls.

**Outcome:** selected MCP tools participate in the same registry, policy, event, budget, and UI contracts as built-ins.

**Scope:** begin with one transport and one configured server; add per-server client ownership, discovery, schema sanitization, namespacing, redacted configuration, policy integration, timeout/cancel, and cleanup.

**Acceptance criteria:** discovered tools cannot override built-in names; invalid schemas are rejected; no server is trusted by default; filtered secrets do not enter events; cancellation closes the owned client/process; MCP events and approvals replay exactly like built-in tools.

**Focused tests:** namespace collision, malformed schema, malicious descriptions, environment filtering, disconnect, timeout, cancellation, oversized output, server restart, per-server isolation.

**Runtime scenario:** connect one disposable local stdio server exposing a read-only echo tool, approve once, deny once, cancel once, and verify cleanup and persisted events.

**Rollback boundary:** remove the server configuration and unregister the adapter; no built-in tool or event schema changes are reverted.

**Risk:** MCP multiplies external authority and schema quality problems. Add servers/transports one work unit at a time.

## Cross-cutting acceptance scenarios

These scenarios must remain green after the phase that introduces them:

| Scenario | Expected invariant |
|---|---|
| Repository file contains prompt injection telling the model to disable policy | Text may affect the model proposal; policy authority and capabilities remain unchanged. |
| Model emits explanatory `bash` fence | It is rendered as text and never becomes a call. |
| Model emits malformed native arguments | Handler invocation count is zero; one structured validation result is recorded. |
| Side-effecting call is proposed or approval is replayed | A fresh approval is bound to exactly one immutable action-plan digest and token; replay or any bound-fact change invokes no handler. |
| File identity/content changes before the immediate commit check | Direct apply aborts, the temporary file is removed, and no partial original is written. |
| Injected path, symlink, parent, or mount change is visible at a defined verification boundary | The plan aborts and the outside sentinel remains unchanged; this proves detection of observed races, not impossible atomic CAS against a hostile final-window swap. |
| Strict mode lacks enforced non-movable ancestry, or a required primitive/stable-workspace precondition is unavailable | Direct apply is refused; only an isolated reviewed patch may be exported for manual/out-of-band application, and status remains `exported`, never `executed`. |
| Prompt snapshot A is rolled back after activating B | Stored compiled/source bytes and digests validate, activation is atomic and audited, and restart selects A; a corrupt snapshot leaves B active. |
| User denies or cancels | The call receives one final result and is not retried. |
| Provider repeats identical call | Duplicate budget terminates predictably. |
| Terminal receives ANSI escape text from model/tool | Renderer neutralizes control effects. |
| Process spawns a child and times out | Entire owned process group terminates and cleanup is recorded. |
| Context compacts across tool work | Call, policy, approval, and result remain one causal unit; source events remain intact. |
| Process crashes after execution starts | Resume reports an indeterminate/pending state and asks for a user decision; it never re-executes silently. |

## Primary risks and mitigations

| Risk | Mitigation |
|---|---|
| Small local models do not reliably emit native tool calls | Measure exact model tags, improve generated tool descriptions and examples, and keep an explicit provider compatibility adapter behind the same admission path. Never re-enable code-fence execution. |
| Security scope is overstated | Distinguish exact action authorization, cooperative direct-apply checks, and hostile-namespace confinement in types and UI. State that the lock is not a security boundary; fail strict mode to reviewed patch export unless OS-enforced non-movable ancestry is proven. |
| Roadmap becomes a rewrite with no usable checkpoints | Each work unit leaves a runnable read-only or diagnostic path and has an independent rollback boundary. |
| Event schemas become unstable | Version payloads, add migrations and projection rebuild tests, and freeze domain event names before TUI polish. |
| TUI work couples rendering back into execution | UI consumes engine events and submits user decisions; it never dispatches a handler directly. |
| Editable prompt reduces tool accuracy | Persist prompt digest, validate size/structure, keep generated tool docs authoritative, and compare through the evaluation matrix. |
| Rich and `prompt_toolkit` fight for terminal ownership | Only `prompt_toolkit` writes to the live terminal; Rich renders in memory. |
| Approval fatigue encourages unsafe auto-accept | Keep fresh exact-plan approval mandatory for every side effect in this product version; improve preview clarity and keystroke flow instead. Remembered grants require a future explicit product decision and threat-model review. |
| SQLite corruption or migration error loses personal history | Back up before migration, retain legacy DB, make projections rebuildable, and never delete source events during compaction. |
| Work units exceed review capacity | Forecast authored line count before implementation and split on the declared rollback boundaries; keep code/tests/docs together. |

## Non-goals

- Public product distribution, plugin marketplace, hosted accounts, telemetry, billing, or multi-user isolation.
- Windows/macOS support or terminal parity beyond the owner's verified Linux environment.
- Cloud-provider compatibility, Gemini-shaped internal types, or provider abstraction for its own sake.
- Training/fine-tuning models or claiming “elite” quality from prompt wording alone.
- A general-purpose secure shell sandbox implemented with command regexes.
- Autonomous root/sudo operations, unattended destructive changes, or hidden background agents.
- MCP before the built-in tool engine and evaluation thresholds pass.
- Copying Easy LLM CLI's React/Ink implementation, branding, or exact visuals.
- Preserving unsafe legacy behavior merely because an existing test expects it.
- Parallel tool execution until identity, policy, budgets, event ordering, and cancellation are proven sequentially.

## Decision log

| ID | Date | Decision | Rationale | Revisit trigger |
|---|---|---|---|---|
| D-001 | 2026-07-31 | Evolve the existing Shadow Code project rather than restart. | CLI, tools, SQLite, UI concepts, and tests provide useful material, while the core can be replaced behind work-unit boundaries. | A baseline audit shows reuse costs more than isolated replacement. |
| D-002 | 2026-07-31 | Keep Python and Ollama first. | Matches the existing code and owner's local models without importing a Node/Gemini runtime. | The owner explicitly adopts another primary runtime/provider. |
| D-003 | 2026-07-31 | Use Easy LLM CLI only as an interaction/architecture reference. | Its persistent history, tool groups, approvals, footer, and responsive rendering are valuable; its Gemini/React/Ink implementation is not. | None; individual patterns may still be rejected through usability evidence. |
| D-004 | 2026-07-31 | Put authority in `PolicyEngine`, not the system prompt. | Prompt text is editable and model-following is probabilistic. | Never for the fundamental boundary; policy rules may evolve. |
| D-005 | 2026-07-31 | Make prompt layers visible, editable, and rollback-capable through immutable byte-complete snapshots. | Digests prove integrity but cannot restore content; compiled/source bytes plus provenance and atomic activation give the owner reproducible recovery. | Prompt compilation or snapshot retention harms measured reliability, with migration preserving existing snapshots. |
| D-006 | 2026-07-31 | Use Pydantic v2-backed `ToolSpec` as the single tool declaration. | Generates schemas and enforces runtime types from one source. | Dependency cost or model-schema incompatibility is demonstrated by tests. |
| D-007 | 2026-07-31 | Remove executable Markdown/code-fence fallback. | Presentation syntax cannot safely establish execution intent. | Never; compatibility must use an explicit provider adapter and the same gates. |
| D-008 | 2026-07-31 | Use append-only SQLite events as authority. | Enables audit, deterministic resume, projections, and safe compression locally. | Proven performance/storage problems on the owner's workload. |
| D-009 | 2026-07-31 | Compact complete causal event ranges and retain originals. | Prevents protocol corruption and makes summaries reversible. | Storage-retention requirements change explicitly. |
| D-010 | 2026-07-31 | Let `prompt_toolkit` exclusively own the terminal; use Rich in memory. | Avoids terminal contention while retaining high-quality Markdown/diff rendering. | A prototype fails on the owner's terminal and a simpler single-renderer design proves better. |
| D-011 | 2026-07-31 | Keep tool execution sequential initially. | Simplifies event ordering, approvals, cancellation, and small-model reasoning. | Sequential invariants pass and a measured workload justifies parallelism. |
| D-012 | 2026-07-31 | Defer MCP until built-in engine evaluation passes. | External servers multiply authority and failure modes. | WU-11 release thresholds pass. |
| D-013 | 2026-07-31 | Require a fresh exact-action approval for every side effect. | Approval is meaningful only when bound one-to-one to immutable arguments, authority, preview, and execution facts; remembered grants are deferred. | A separate owner-approved product change supplies threat-model and evaluation evidence. |
| D-014 | 2026-07-31 | Limit direct mutation to a cooperatively locked, stable workspace; use strict patch export otherwise. | Descriptor-relative/no-follow operations and immediate checks reject observed drift, but directory FDs follow renamed objects and replacement is not atomic prior-inode/digest CAS against a hostile same-UID mutator. | An OS-enforced non-movable-ancestry primitive is adopted and proven before strict direct apply is enabled. |

## Start condition

Implementation begins with **WU-00**, immediately followed by **WU-01**. Do not start visual redesign, prompt expansion, autonomous shell execution, or MCP first. The first usable milestone is a read-only agent path where native model calls are typed, workspace-contained, policy-admitted, event-ready, and incapable of executing an explanatory code fence.
