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
effects (`bash`) require an interactive one-shot approval: Shadow Code shows the exact
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
/compact       Manually compact conversation
/history       Show recent messages
/save [name]   Save session
/load [id]     Load session
/list          List saved sessions
/skills        List available skills
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
- Default model: `shadow-gemma:latest` (configurable via `SHADOW_MODEL` env var)

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SHADOW_MODEL` | `shadow-gemma:latest` | Ollama model to use |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API URL |
| `SHADOW_LEGACY_MARKDOWN_TOOLS` | disabled | Opt in to the unsafe compatibility tool path |

## Architecture

```
shadow-code/
  shadow_code/
    main.py           Entry point, REPL, context management
    prompt.py          System prompt (Claude Code adapted, 17K chars)
    parser.py          Legacy Markdown tool call detection
    ollama_client.py   Ollama API streaming client
    conversation.py    Message history, 3-tier context management
    display.py         Streaming buffer (hides tool JSON from user)
    compaction.py      LLM-based conversation summarization
    skills.py          Skill system (/commit, /review, etc.)
    safety.py          Legacy destructive command detection
    ui.py              Rich terminal rendering
    streaming.py       Rich Live streaming display
    repl.py            prompt_toolkit REPL with history
    db.py              SQLite session persistence
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

## License

[MIT](LICENSE)
