# Product

Shadow Code is a private, local-first coding agent for its owner’s personal Linux and Ollama environment. It exists to make model-driven coding trustworthy: capable native tool use, visible intent, controlled side effects, recoverable sessions, and complete ownership of the system prompt.

## Register

product

## Users

The sole user is the owner, working locally on Linux with Ollama. The primary workflow is an ongoing terminal-based coding session in which the agent understands a task, selects and calls tools correctly, explains consequential actions, and helps modify or inspect local projects without surrendering control of the machine or prompt.

## Product Purpose

Shadow Code should provide the reliability and interaction quality expected from an expert coding agent while remaining fully personal and locally operated.

Its core promises are:

- Native, schema-valid tool calling rather than executable prose or inferred code blocks.
- Explicit approval before any side effect, with the proposed action visible first.
- Bounded execution that cannot enter an uncontrolled tool loop.
- Durable, replayable sessions that recover from interruption through an authoritative event log.
- System prompt sovereignty: the owner can show, edit, diff, validate, reload, and roll back the active prompt.
- Exact optimization and evaluation for `gemma4:e4b-it-qat` and `granite4.1:8b` in the owner’s real environment.

## Brand Personality

**Calm, precise, trustworthy.**

Shadow Code communicates with expert restraint. It makes state, intent, risk, and failure understandable without theatrics. Confidence comes from evidence and predictable behavior, not decoration or exaggerated language.

## Anti-references

- No noisy raw-debug transcript presented as a finished interface.
- No decorative, web-like, or visually overloaded terminal UI.
- No hidden execution, ambiguous progress, or activity implied by animation alone.
- No imitation of Easy LLM CLI’s branding or implementation. Its interaction quality is a clean-room reference; Shadow Code retains a distinct identity and architecture.
- No interface element that competes with the coding task for attention.

## Design Principles

### Trust is observable

Before consequential work, show what will happen and wait for approval. During work, expose the real lifecycle. Afterward, report the actual result. Never imply that an action ran unless a validated tool call was executed and recorded.

### The owner remains sovereign

System prompts, permissions, session history, and tool behavior must be inspectable and controllable. Customization may change agent behavior, but it must never silently weaken the execution boundary.

### Correctness outranks immediacy

Reject malformed calls, surface actionable errors, and stop bounded loops rather than guessing. A clear refusal or recoverable failure is better than an unsafe approximation.

### The interface disappears into the task

Use familiar terminal interactions, stable placement, concise status language, and progressive disclosure. Preserve context and flow instead of making the owner decode the interface.

### Local reality is the benchmark

Design and evaluate against the owner’s actual Linux environment, Ollama runtime, models, terminal constraints, and coding workflows—not generic provider assumptions.

## Accessibility & Inclusion

- The interface must remain usable in narrow terminals and degrade without losing essential information.
- Every state uses explicit text plus a glyph; color is never the sole signal.
- Unicode has a complete ASCII fallback.
- `NO_COLOR` is respected without reducing clarity or capability.
- Motion is minimal, purposeful, and limited to communicating active state; equivalent static feedback remains available.
- Approval, error, cancellation, and recovery states use unambiguous language and consistent controls.

## Success Criteria

- Rendered Markdown, code fences, and assistant prose can never trigger execution; only validated native tool-call events can.
- Every tool call is checked against its authoritative schema before execution. Invalid calls produce no side effect.
- Every side effect is previewed and explicitly approved before it runs, and the decision is recorded.
- Turn, time, tool-call, and output budgets terminate runaway or repeated behavior predictably.
- An interrupted session can be reconstructed from the event log without inventing messages, approvals, calls, or results.
- The active compiled system prompt can be shown, changed, diffed, validated, reloaded, and rolled back, with its identity recorded per request.
- Recorded and live evaluation suites verify tool selection, argument accuracy, multi-step execution, malformed-call recovery, denial and cancellation, loop termination, and security boundaries on both target models.
- The terminal UI preserves clear task state and controls in narrow, Unicode, ASCII, and `NO_COLOR` modes.

## Privacy Boundary

Shadow Code is offline-first and privacy-first. Prompts, source code, tool arguments, outputs, session events, and configuration remain on the owner’s machine by default. Local Ollama is the default and authoritative model path. Network access is not implicit: any network-capable tool or provider must be deliberately configured, visibly identified, and approved before data leaves the local boundary.

The execution policy is independent of the editable system prompt. Prompt changes cannot grant filesystem, shell, network, or MCP capabilities by themselves.

## Non-goals

- Public distribution, multi-user administration, hosted accounts, telemetry, or cloud synchronization.
- Cross-platform parity or generic support for environments other than the owner’s Linux system.
- Recreating Easy LLM CLI, Gemini CLI, Cline, or another agent’s internal architecture or visual identity.
- Treating prompt engineering as a substitute for schema validation, policy enforcement, sandboxing, or recovery.
- Maximizing autonomy at the expense of visibility, approval, bounded execution, or local privacy.
