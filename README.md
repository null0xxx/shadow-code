# shadow-code

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Ollama-local%20LLM-orange.svg)](https://ollama.com)

> Local Ollama chat assistant with an in-progress, policy-controlled coding runtime

## Run the current checkout

From the repository root, use the verified local environment and model:

```bash
SHADOW_MODEL=gemma4-cline:32k .venv/bin/shadow-code
```

When startup succeeds, Shadow Code displays this prompt:

```text
shadow>
```

You can now ask questions, explain code you paste into the conversation, and use the
conversation and session commands.
The safe default is **chat-only**: native tool calls fail closed and execute nothing until
the admission and approval wiring is complete.

## Install and run elsewhere

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com), running locally
- An Ollama model already pulled, for example `gemma4-cline:32k`

Install Shadow Code in a virtual environment:

```bash
python -m venv .venv
.venv/bin/pip install -e '.[full]'
```

Start it with the name of a model available in Ollama:

```bash
SHADOW_MODEL=<ollama-model-name> .venv/bin/shadow-code
```

If Ollama is not available at `http://localhost:11434`, set `OLLAMA_HOST`:

```bash
OLLAMA_HOST=http://localhost:11434 \
SHADOW_MODEL=<ollama-model-name> \
.venv/bin/shadow-code
```

Alternatively, `scripts/install.sh` installs (or updates) a dedicated venv
under the XDG data dir and puts a `shadow-code` launcher on `~/.local/bin`;
see *Operational polish (WU-12)* below for uninstall and purge.

## Current capabilities

Shadow Code sends your messages to the selected local Ollama model and streams its text
responses. The current default runtime supports:

- local chat without API keys or cloud model calls;
- conversation history, context management, and session persistence;
- slash commands such as `/help`, `/save`, `/load`, and `/compact`;
- optional Rich and prompt-toolkit terminal presentation.

### Native tool execution and approval

Read-only native tool calls (`read_file`) execute through the admission pipeline: registry
validation, then policy, then a contained executor. Calls with side effects or unknown
effects (`bash`, `write_file`, `edit_file`) require an interactive one-shot approval:
Shadow Code shows the exact
action plan (tool, version, capability, arguments, workspace, digest) and executes only
after an explicit `y`. The approval token is bound to the action-plan digest, authorizes
exactly one execution, and is burned by any mismatch — changed arguments, workspace, tool
version, or registry reject it. Denial or cancellation is final and the call is not
retried.

`bash` executes the approved command UNCONFINED — no sandbox is applied, so the
approval is the only control. The approval plan binds the exact command, the
workspace, the process-environment digest, the shell resolution, and the sandbox
facts; the preview visibly labels `sandbox: unconfined` (noting when a sandbox
helper is detected on the host but not applied, since commands are never wrapped)
and lists detected shell
features (substitution, redirection, pipes, chains, backgrounding) before you
approve. Execution runs in its own process group with a predictable timeout: on
timeout or cancellation the whole group is terminated, and truncated output
records how many bytes were removed. The child process receives a minimal
allowlisted environment (`PATH`, `HOME`, `LANG`, `TERM`, and similar) — parent
secrets such as API keys and tokens are never passed through.

Strict mode denies shell execution entirely when no kernel sandboxing
(`bwrap`/`firejail`) is available on the host:

```bash
SHADOW_BASH_STRICT=1 .venv/bin/shadow-code
```

### Agent engine (bounded turns)

Multi-step tool work is driven by a bounded engine (`shadow_code/engine.py`) with
explicit states — `STREAMING`, `COLLECTING`, `ADMITTING`, `AWAITING_APPROVAL`,
`EXECUTING` — and exactly one terminal outcome per turn: `COMPLETED`, `CANCELLED`,
`FAILED`, or `BUDGET_EXHAUSTED` with a typed reason (`budget_steps`,
`budget_calls`, `budget_time`, `budget_output`, `duplicates`). The engine never
prints or reads input: streaming, approval consent, cancellation, and round
reporting are injected seams, so the same admission pipeline (registry
validation, policy, one-shot approval, contained execution) runs without any UI
dependency. Every proposed call reaches exactly one terminal result, a denial is
final and never retried, and a transient provider failure retries the round
exactly once before failing.

Per-turn budgets bound the work: native rounds (default 4), total calls, wall
clock, and aggregate result output. The same call (name + canonical arguments)
may repeat at most twice per turn — a third occurrence is not executed and ends
the turn as `duplicates`. Cancellation is checked before every state
transition: no further handler runs once requested, and a `KeyboardInterrupt`
inside a handler (which already kills its process group) becomes a `CANCELLED`
outcome, never a silent re-execution.

### File mutations (`write_file` / `edit_file`)

File changes go through the same admission pipeline as every other call — there is no
append or write path that bypasses read, preview, and policy. `write_file` replaces a
file's full content; `edit_file` replaces exact text and fails closed unless the text
matches EXACTLY ONCE (zero matches report `no_match`, duplicate matches report
`ambiguous_match` and nothing is written). Both require a fresh interactive one-shot
approval bound to the exact arguments and a previewed unified diff, and every mutation
consumes its own approval.

An approved mutation is applied atomically under the declared stable-workspace /
cooperative-writer threat model:

- the new content is written to a temp file in the target directory, fsynced, and given
  the existing file's mode (new files get `0o644`); all opens, temp creation, and the
  final rename are descriptor-relative and no-follow beneath the pinned workspace root;
- immediately before the commit, the target is re-snapshotted and compared against the
  approved plan — any change in identity, content, mode, or presence aborts with the
  original intact and the temp file removed;
- the rename plus a directory fsync complete the commit, and a post-write readback
  verifies the landed digest; the result records before/after identities and SHA-256
  digests.

Cooperating Shadow Code writers serialize their commits on a per-workspace lock
(`.shadow-code.lock`). This lock is **cooperative only**: it is honored solely by
Shadow Code writers that voluntarily take it and is **not a security boundary** — it
offers no protection against uncooperative or hostile processes.

Strict mode refuses direct application entirely — because no enforced,
non-movable workspace ancestry primitive is adopted (roadmap D-014), no
atomic guarantee against a hostile final-window swap is claimed. Instead of
denying the capability, strict mode keeps `write_file`/`edit_file` available
but routes every approved change to a **patch export**: the full unified diff
is written to `<workspace>/.shadow-code-exports/` and reported with status
`exported`. The workspace target is never touched on this path — the export
is a reviewed-patch fallback, not confinement:

```bash
SHADOW_MUTATION_STRICT=1 .venv/bin/shadow-code
```

### Legacy Markdown tools (compatibility only)

An older Markdown-based tool protocol can be enabled explicitly:

```bash
SHADOW_MODEL=gemma4-cline:32k \
SHADOW_LEGACY_MARKDOWN_TOOLS=1 \
.venv/bin/shadow-code
```

> **Warning:** Use this mode only in a disposable workspace with no secrets or valuable
> uncommitted changes. It bypasses the new native admission and approval path and can invoke
> file and shell tools. It is a compatibility path, not the production-safe runtime.

### System prompt layers and snapshots

The system prompt is compiled from fixed layers, in this order:

1. **builtin base** — behavioral instructions, owned by the code;
2. **user overlay** — `~/.config/shadow-code/prompt.md` (optional, appended with a
   provenance header);
3. **workspace overlay** — `<workspace>/.shadow-code/prompt.md` (optional);
4. **generated tool documentation** — always rendered last from the live tool registry.

Layer 4 is never editable: the registry stays the single source of tool truth, and prompt
text can never grant a capability — policy decisions never read prompt contents.

Every compilation is deterministic (identical inputs produce identical bytes and digest)
and is stored content-addressed under `~/.local/state/shadow-code/prompts/<digest>/`
together with its normalized source bytes, so any snapshot reproduces itself exactly
without consulting current files. Startup and every switch print `prompt snapshot:
<digest>`; switching prints an audit line `prompt: active <old> -> <new>`. The active
pointer changes atomically, and only after the target snapshot has been re-verified — any
failure leaves the previous active untouched.

Overlays are watched per turn: editing one affects only the next turn (recompile, save,
activate). Use `/prompt` to manage this:

```
/prompt show                Print the active compiled prompt (first 200 lines)
/prompt sources             List layers with origin, sha256, and size
/prompt history             List snapshots (digest, timestamp, layers)
/prompt diff [digest]       Diff a previous snapshot against the active one
/prompt validate            Check structure, digests, and tool-doc freshness
/prompt reload              Force recompile + activate
/prompt edit                Open the user overlay in $EDITOR, then reload
/prompt rollback <digest>   Verify + validate a snapshot, then activate it atomically
```

A rollback is verified against stored bytes and the live registry before activation; if
anything fails, the active prompt stays unchanged. Rollback pins the snapshot for the
rest of the session — the next startup recompiles from whatever sources exist then.

### Event store and resume

Every causal action of a turn is appended to an append-only event log at
`~/.local/state/shadow-code/events.db` (SQLite, WAL). Recorded events: session start and
end, user and assistant messages, each proposed tool call, the policy decision, approval
requests/grants/denials bound to the action-plan digest, the terminal tool result, and a
per-turn `turn_completed` carrying the active prompt snapshot digest. Causally linked
events land in a single transaction, duplicate appends are idempotent, and there is no
update or delete API.

On startup, if the most recent session has proposed calls with no terminal result, the
CLI prints the pending call ids, tool names, and plan digests and asks before continuing:
continuing abandons the pending work and starts a fresh session — nothing is ever
re-executed on resume. `/events` runs integrity diagnostics for the current session
(sequence contiguity, payload versions, call-id references). The event store never
breaks the CLI: if it cannot be opened or written, a warning is printed and the session
continues without events.

Legacy sessions from `~/.shadow-code/sessions.db` are never migrated. To audit one
through the event store, call `EventStore.import_legacy_session(path, session_id)` —
it copies the messages read-only into a `legacy-<id>` event session and is idempotent.

### Context compaction (WU-08)

Long sessions shrink model context without losing causal tool groups or source
history. The event log is the source of truth and is **never modified** by
compaction — no event is ever updated or deleted.

- **Causal groups.** The event stream is grouped into indivisible units: a tool
  call's proposal → policy decision → approval → result chain is one group, a
  plain message is a singleton. A proposed call without a terminal result is
  pending and can never be selected.
- **Closed-range selection.** `/compact` selects the oldest complete groups that
  fit the context budget. A selection never splits a group, never takes a
  pending one, and stops at the first pending group; a multi-call batch whose
  proposals landed together is all-or-nothing.
- **Snapshots.** The selected range is summarized by the model into a typed
  `context_snapshot` event. The snapshot records the covered sequence range,
  group count, a digest of the covered event ids, and a source digest that also
  binds the projection-logic version — changed projection logic or tampering is
  detected on validation. Validation also flags hallucinated file references:
  paths that appear neither in the covered payloads nor beneath the workspace.
  A failed summary or failed validation creates **no** snapshot and leaves the
  conversation untouched.
- **Projection.** After compacting, the provider sees one synthetic assistant
  message with the summary plus the projection of everything after the covered
  range, so tool results are never orphaned. The original events remain
  queryable at any time (`/events`, `events_for`).
- **Emergency reduction** (`emergency_reduce`) is projection-level only: beyond
  the snapshot it drops the oldest complete terminal groups in favor of a
  placeholder marker and never touches pending groups; the event log stays
  intact.
- **Diagnostics.** `/context` prints group counts by kind, terminal/pending
  counts, uncovered token estimate, active snapshot coverage and digests, and
  integrity issues.

The legacy 3-tier message-level compaction remains as the automatic safety net
and as the fallback when the event store is unavailable.

### Persistent TUI (opt-in, WU-09)

Set `SHADOW_TUI=1` to run a persistent terminal shell instead of the
line-oriented REPL. The TUI only activates when stdin/stdout are real TTYs and
prompt_toolkit is installed; anything missing falls back to the line REPL,
which doubles as the minimal diagnostic client (the roadmap rollback boundary:
removing the TUI changes neither execution nor stored history).

```
SHADOW_TUI=1 shadow-code
```

Layout: a read-only **transcript** (your messages, assistant text streamed
live as it arrives, tool lifecycle groups, budget and slash-command output),
a bounded **approval panel** that appears only while a decision is pending,
a **multiline input** area, and a **footer** showing
model, token usage, context %, workspace root, active prompt snapshot digest,
and permission labels (e.g. `bash:UNCONFINED`, `mutation:export`).

Keybindings:

```
Enter          Send (a no-op while an approval is pending)
Alt+Enter      Insert a newline (Esc then Enter)
y / n          Approve / deny while the approval panel is focused
Esc            Deny a pending approval (same as n)
Ctrl+E         Expand/collapse the latest truncated tool output
Ctrl+C         Cancel the active turn (denies a pending approval)
Ctrl+D         Exit (denies a pending approval, cancels active work first)
Ctrl+U         Clear the input
Ctrl+L         Repaint
Up/Down        History (scroll the approval panel while it is pending)
```

### Tool lifecycle and approvals (WU-10)

Tool calls render **grouped per engine round** under a `step n` header, each
row carrying a status token that updates in place as the call moves through
`proposed → awaiting approval → executing → ok | denied | failed | exported`:

```
step 1
  ok read_file  target.txt  [ok]
  ok write_file  target.txt  [ok]
    mutation: write target.txt
    status: executed
  x bash  $ touch bash-ran.txt  [denied — not retried]
    hint: denied by user — not retried
```

Unicode glyphs (`✓ ○ ✗ …`) are decorative; the text label is the evidence and
survives every theme. ASCII/NO_COLOR mode uses `ok/o/x/...` markers instead.
Approved mutations in strict mode are labeled `[exported]` (patch written to
`.shadow-code-exports/`, workspace untouched) versus `status: executed` in the
result body of an applied mutation. Failed and denied rows get a one-line
`hint:` mapped from the typed error code (`no_match` → "re-read the file; the
exact text changed", `workspace_drift` → "file changed on disk; re-read and
retry", …).

Results longer than 12 lines collapse to head + tail with a
`… M more lines (Ctrl+E: expand)` marker; `Ctrl+E` toggles the full content in
place — the transcript scrolls, the input area never moves. Long single lines
(huge paths, huge commands) wrap instead of being truncated.

Approvals keep the fail-closed contract, now through a **single-focus panel**
rendered above the input area when the engine enters AWAITING_APPROVAL. The
panel shows every approval-bound action-plan fact before you decide: tool and
version, capability, full canonical arguments, workspace identity, registry
digest, plan digest, execution facts, and the exact preview (a unified diff
for mutations; command + sandbox + feature classification for bash). The
panel is bounded (max 14 rows) and scrolls with Up/Down; the input area is
unreachable while it is focused. Only `y` approves; `n`, Esc, Ctrl+C, and
Ctrl+D all deny. Each side-effecting call in a batch gets its OWN fresh
control — there is no "approve all" and consent never carries over between
calls. A denial is final and stays visible as a `denied — not retried` row;
a stale plan (authority-fact change before execution) surfaces as
`plan changed — approval rejected`; a file changed on disk between preview
and execution surfaces as the `workspace_drift` failure with its hint.

Cancelling mid-turn sets the
engine's cancel flag — the turn ends as CANCELLED and prompt_toolkit restores
the terminal, never leaving it corrupted.

The footer condenses across width buckets (≥120, ≥80, ≥40, <40 columns);
input content and focus survive resizes. Model and tool output is sanitized
before rendering, so no ANSI/control sequence can escape into your terminal.
`NO_COLOR` disables all styling and forces plain-ASCII decoration (which
`SHADOW_ASCII=1` also enables on its own); every semantic label stays visible
as text.

In TUI mode Rich performs no direct terminal writes — prompt_toolkit owns the
screen. The engine, event store, policy, and provider layers are untouched;
the TUI drives the exact same session machinery through injected seams.

### Evaluation matrix (WU-11)

Tool-calling quality is **measured on the exact installed models**, not
inferred. The matrix has three layers:

- **Versioned scenario corpus** (`shadow_code/eval/corpus/v1/*.json`, one
  file per scenario). The corpus is pure data and runs **unchanged** for
  every model family — no model-specific fields or special-casing. v1 holds
  15 scenarios: 8 `safety_invariant` (prompt injection in repository
  content, symlink escape, denied command, cancellation, malformed native
  call, repeated-call termination, strict patch export, stale
  preview/concurrent-mutation drift) and 7 `capability` (read-only
  orientation, targeted read, exact edit, new file, focused test, multi-step
  failure recovery, context-compression continuation). Each scenario
  declares its workspace, prompt, structured expectations, and the scripted
  transcript the deterministic suite replays.
- **Deterministic suite** (always on, CI):
  `tests/unit/test_eval_corpus.py` validates the corpus (unique ids, schema,
  all 15 required ids, thresholds file); `tests/unit/test_eval_deterministic.py`
  drives every scenario's transcript through the **real** engine, registry,
  policy, guard, and mutation paths in tmp workspaces and asserts the
  invariant directly — zero handler runs for malformed calls, sentinel
  unchanged, denial final and never executed, duplicate-budget termination,
  exported-not-applied patches, drift aborts.
- **Live harness** (opt-in, removable; CI never calls Ollama):
  `python -m shadow_code.eval --model gemma4-cline:32k [--scenario id]
  [--report path]` runs the corpus against a real local Ollama in
  **disposable temporary workspaces only** (deleted after every scenario).
  Consent **denies by default**; capability scenarios carry
  `auto_approve: true` in the corpus. Engine-facing invariants whose trigger
  a well-behaved model never produces (malformed calls, duplicate loops) run
  with `live_driver: "transcript"`. The harness measures — it never weakens
  validation, budgets, or approvals to fit an outcome.

Per scenario the harness scores tool choice, argument validity, path
accuracy, calls-to-completion, duplicate rate, denial compliance, malformed
recovery, budget adherence, edit correctness, verification honesty, latency,
and peak context, and classifies failures into a stable taxonomy
(`wrong_tool`, `invalid_args`, `path_miss`, `no_completion`,
`duplicate_loop`, `denial_violation`, `malformed_recovery`,
`budget_violation`, `edit_incorrect`, `dishonest_verification`,
`injection_breach`, `containment_breach`, `export_violation`). Every score
links to the **redacted** raw events digest (no absolute home/tmp paths, no
environment values) plus the exact model tag, prompt digest, registry
digest, and corpus version. Reports land in `eval/reports/`
(gitignored); `shadow_code.eval.report.compare(a, b)` renders a two-model
text table.

Release thresholds are **declared before any tuning** in
`shadow_code/eval/thresholds.json` and travel inside each report: safety
invariants require a 100% pass rate, capability 60% — and any
`safety_invariant` failure **blocks release regardless of the aggregate
score**.

### Operational polish (WU-12)

Startup prints a short **authority block** naming every active authority
boundary and every unavailable capability before the first prompt: the
contained workspace root with its device/inode identity and containment
mode, the granted capabilities, each withheld capability with its reason
(`network.access` is never granted by this build; `mcp.invoke` is granted
only when a configured MCP server discovered usable tools — see below;
`process.execute` is withheld under `SHADOW_BASH_STRICT=1` without a kernel
sandbox), the bash sandbox label, the mutation mode, the one-shot approval
model, and the active prompt digest.

Three slash commands cover diagnostics and recovery, in both the line REPL
and the TUI:

- `/doctor` — a redacted diagnostic report: effective configuration with
  per-setting source (`env:` vs `default`) and any corrupt env values that
  fell back to defaults, Ollama reachability with an actionable next step
  (`ollama serve`, `ollama pull <model>`, or the legacy-Markdown
  compatibility path when the model cannot serve native tool calls),
  workspace identity and containment, granted/withheld capabilities, prompt
  snapshot, event-store schema version (including pending-migration and
  newer-than-this-build previews) and integrity, legacy session count, and
  a sweep of post-crash `.shadow-tmp-*` orphans (never while a live
  instance holds the workspace mutation lock, never inside
  `.shadow-code-exports/`).
- `/backup` — copies the session and event databases into a timestamped
  directory under `~/.local/state/shadow-code/backups/` with fsync and a
  `manifest.json` recording size and sha256 per file.
- `/restore [dir]` — shows a **dry-run plan first** (per-database digests,
  what would change), then writes only after an explicit one-shot
  confirmation; without a confirmation seam it fails closed. The apply pass
  re-verifies every backup file against its manifest digest before
  overwriting anything, so a tampered backup refuses with nothing written.
  With no argument the newest backup is used.

Every diagnostic line, receipt, and manifest passes through redaction:
values of secret-looking environment variables (`*TOKEN*`, `*PASSWORD*`,
...) plus the explicit comma-separated `SHADOW_REDACT` list are replaced
with `***` and never reach the terminal or a backup manifest.

`scripts/install.sh` is the single local install/update/uninstall command
(no sudo):

```bash
scripts/install.sh                    # install or update: venv + pip + ~/.local/bin/shadow-code
scripts/install.sh uninstall          # remove executable and venv, PRESERVE user data
scripts/install.sh uninstall --purge  # also delete user data, behind an interactive confirmation
```

Uninstall never deletes `~/.local/state/shadow-code` (sessions, events,
prompt snapshots, backups) or `~/.config/shadow-code` (prompt overlays)
unless `--purge` is given and confirmed by typing `purge`.

### MCP servers (WU-13)

Shadow Code can connect to local [MCP](https://modelcontextprotocol.io)
servers over the **stdio transport** with a hand-rolled, dependency-free
client (newline-delimited JSON-RPC 2.0: `initialize` →
`notifications/initialized` → `tools/list` → `tools/call`). Servers are
configured in `~/.config/shadow-code/mcp.json` (`$XDG_CONFIG_HOME` is
honored) — a file, never environment variables:

```json
{
  "servers": {
    "echo": {
      "command": "/usr/bin/python3",
      "args": ["/path/to/echo_server.py"],
      "env": {"SOME_FLAG": "1"},
      "enabled": true,
      "connect_timeout_seconds": 10,
      "call_timeout_seconds": 60,
      "max_output_chars": 20000
    }
  }
}
```

At startup each enabled server is spawned in its **own process group** with
a minimal allowlisted environment (parent secrets such as API keys are
never inherited; only the allowlist plus the explicitly configured `env`
entries reach the server), initialized, and queried for its tools. Every
discovered tool is sanitized — control characters stripped, descriptions
length-capped, malformed schemas and non-scalar properties rejected with a
visible issue — and registered under a **namespaced** name
`mcp_<server>__<tool>`. Namespacing is structural: a discovered tool can
never shadow a built-in name, and duplicate names are rejected.

**Safety model: no server is trusted by default.** MCP tools register with
capability `mcp.invoke` and unknown side effects, so they flow through the
same registry → policy → approval → executor → events pipeline as built-in
tools and **every call requires a fresh one-shot approval** bound to the
exact action-plan digest — a read-only declaration from the server is never
taken on faith. Timeouts (connect and per-call, configurable per server)
kill the whole owned process group, as do cancellation and disconnect;
oversized output is truncated with a marker instead of crashing. A server
that fails to initialize is reported unavailable at startup and in
`/doctor` (redacted: configured env values are never shown) and never
blocks the session. There is **no auto-reconnect**: once a server process
dies or times out, later calls fail closed with a typed error until the
next session. Session exit closes every owned client and process group and
records each server's lifecycle (`ready`/`unavailable`/`closed`) in the
event store; MCP tool proposals, approvals, and results are the same
versioned event payloads as built-ins and replay identically.

**Limitations (first transport, one work unit):** stdio servers only; only
flat scalar tool arguments (string/integer/number/boolean) are supported —
tools with array/object/nested argument schemas are rejected at discovery;
optional arguments are sent with their type's zero value when omitted.
Removing a server from `mcp.json` (or setting `"enabled": false`) fully
unregisters it; no built-in tool or event schema changes.

## Features

| Feature | Description |
|---------|-------------|
| **Safe default** | Local text chat; native tool requests fail closed without execution |
| **Legacy tools** | Optional compatibility path for bash, file, glob, grep, and directory tools |
| **13 Skill prompts** | `/review`, `/debug`, `/explain`, and more; tool-dependent actions remain unavailable by default |
| **Context Management** | 3-tier: result clearing, LLM compaction, emergency truncate |
| **Session Persistence** | Save/load conversations with SQLite |
| **Georgian + English** | Responds in the language you write in |
| **Rich UI** | Markdown rendering, spinners, color-coded context bar |

## Commands

```
/help          Show all commands
/clear         Clear conversation
/tokens        Show context usage
/info          Session info
/cd [path]     Change working directory
/compact       Compact complete context groups into an event-sourced snapshot
/context       Show context diagnostics (groups, tokens, snapshot coverage)
/history       Show recent messages
/save [name]   Save session
/load [id]     Load session
/list          List saved sessions
/skills        List available skills
/prompt        Inspect, reload, and roll back the system prompt
/events        Verify event store integrity
/doctor        Diagnose config, model, workspace, and stores (redacted)
/backup        Back up the local databases with a manifest
/restore [dir] Preview a database restore, then confirm to apply
/version       Version info
/exit          Exit
```

## Skills

These commands load task-specific prompts. In the safe/default mode, they can guide analysis
and text responses, but commands that require file or shell tools cannot execute those tools.

```
/commit        Create a git commit
/pr            Create a pull request
/review        Review code for bugs and security
/simplify      Review for code quality
/test          Run tests and analyze results
/debug         Debug an error
/explain       Explain code in detail
/refactor      Refactor while preserving behavior
/search        Deep codebase search
/verify        Verify changes actually work
/init          Explore a new project
/remember      Save info for later
/stuck         Get help when stuck
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) with a local model
- Default model: `gemma4-cline:32k` (configurable via `SHADOW_MODEL` env var)

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SHADOW_MODEL` | `gemma4-cline:32k` | Ollama model to use |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API URL |
| `SHADOW_LEGACY_MARKDOWN_TOOLS` | disabled | Opt in to the unsafe compatibility tool path |
| `SHADOW_BASH_STRICT` | disabled | Deny shell execution when no kernel sandbox is available |
| `SHADOW_MUTATION_STRICT` | disabled | Export approved file changes as reviewed patches under `.shadow-code-exports/`, never apply them |
| `SHADOW_TUI` | disabled | Opt in to the persistent TUI (real TTYs + prompt_toolkit required) |
| `SHADOW_ASCII` | disabled | Plain-ASCII TUI decoration (`NO_COLOR` forces this too) |
| `SHADOW_REDACT` | empty | Comma-separated extra values redacted from `/doctor`, `/backup`, and `/restore` output |

## Architecture

```
shadow-code/
  shadow_code/
    main.py           Entry point, REPL, context management
    prompt.py          System prompt (Claude Code adapted, 17K chars)
    parser.py          Legacy Markdown tool call detection
    provider.py        Provider-neutral typed streaming contract
    ollama_client.py   Ollama API streaming client (thin adapter over provider.py)
    conversation.py    Message history, 3-tier context management
    context_compaction.py  Event-sourced causal groups, snapshots, projection
    display.py         Streaming buffer (hides tool JSON from user)
    compaction.py      LLM-based conversation summarization
    skills.py          Skill system (/commit, /review, etc.)
    safety.py          Legacy destructive command detection
    ui.py              Rich terminal rendering
    streaming.py       Rich Live streaming display
    repl.py            prompt_toolkit REPL with history
    db.py              SQLite session persistence
    ops.py             /doctor diagnostics, DB backup/restore, crash recovery (WU-12)
    tool_context.py    Shared state (CWD, read files)
    tools/             Legacy compatibility tool implementations
      bash.py          Shell commands with CWD tracking
      read_file.py     File reading with line numbers
      edit_file.py     Exact string replacement
      write_file.py    File creation/overwrite
      glob_tool.py     File pattern matching
      grep_tool.py     Content search (rg -> grep -> python)
      list_dir.py      Directory listing
```

## How It Works

1. **System prompt** tells the LLM about coding practices and the currently enabled protocols
2. **Native tool calling** runs read-only tools through policy; side-effecting calls need a one-shot digest-bound approval
3. **Legacy tool calling** uses ` ```tool_call ` Markdown only when explicitly enabled
4. **Context management** follows Claude Code's pattern: clear old results at 55%, LLM summarization at 65%, emergency truncate at 85%
5. **KV cache** optimization: system prompt is 100% static for Ollama cache hits

## Provider Contract

`provider.py` turns Ollama's NDJSON stream into one typed event stream —
`TextDelta`, `ToolCallStarted`, `ToolCallArgumentsDelta`, `ToolCallComplete`,
`UsageUpdate`, `TurnDone`, `ProviderError` — independent of UI and engine.
Raw provider dictionaries never leave the module.

- **Fail-closed normalization:** fragmented argument strings accumulate per
  call and parse only when the call completes; unparseable arguments are
  carried raw so registry validation rejects them. Partial calls never execute.
- **Order and identity:** calls keep provider IDs (generated `call-<n>` when
  missing) and complete in stream order.
- **Typed errors:** HTTP failures, timeouts, disconnects, and malformed
  payloads terminate the stream with a coded `ProviderError`.
- **Cancellation and timeouts:** closing the stream closes the HTTP response
  and stops the reader thread; connect/read timeouts are configurable.

`ollama_client.py` is a thin sync adapter over this contract; its public
surface (`chat_stream`, `last_tool_calls`, token tracking) is unchanged.

## License

[MIT](LICENSE)
