# Shadow Code -- Final Implementation Plan

> Version: 4.0 | Date: 2026-04-05
> Status: FINAL (5-Eye Team Reviewed, Challenge v2+v3 all fixes applied)
> History: v1 -- Challenge -- v2 -- Challenge v2 -- v3 -- Challenge v3 -- this v4

---

## Changes from v1

| Issue | v1 Problem | v2 Fix |
|-------|-----------|--------|
| C1 | Prompt had placeholders | Full prompt text word-for-word |
| C2 | Tool call format untested | Phase 0: empirical validation |
| C3 | Streaming shows raw JSON | Streaming buffer hides tool_call |
| C4 | Token estimation unreliable | Ollama prompt_eval_count based |
| C5 | Tool results confuse model | Context prefix added |
| C6 | No Phase 0 | Phase 0: Validation added |
| C7 | shell=True undocumented | Intentional + interactive detect |
| H1 | CWD doesn't change with cd | CWD tracking in bash.py |
| H5 | edit_file allows blind edits | File read tracking |
| H9 | No few-shot examples | 2 full examples in prompt |

---

## 1. Verified Facts (shadow-gemma:latest)

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | Gemma 3, 27.4B, Q4_K_M, ~17GB | ollama show |
| Context window | 131,072 (128K) | ollama show |
| Native tool calling | NO | API test: "does not support tools" |
| JSON output | Yes | Tested |
| Template | start_of_turn user/model | Modelfile |
| Stop token | end_of_turn | Modelfile |
| Temperature | 0.9 (default) | Modelfile |
| API | http://localhost:11434/api/chat | standard |

---

## 2. Architecture

```
User Input
    |
    v
main.py (REPL + signal handling)
    |
    v
ollama_client.py ----> Ollama API (streaming)
    |
    v
display.py (streaming buffer: hides <tool_call> JSON, shows [tool] info)
    |
    v
parser.py (regex + JSON validation)
    |
    v
tools/ (registry + dispatch + execute)
    |
    v
conversation.py (history + token tracking + truncation)
    |
prompt.py (system prompt - FULL TEXT, no placeholders)
```

### File Structure

```
/home/n00b/makho/shadow-code/
  pyproject.toml
  shadow_code/
    __init__.py
    main.py              # REPL, signal handling, startup health check
    config.py            # Constants, env var overrides
    prompt.py            # STATIC system prompt (no dynamic content!)
    parser.py            # Tool call regex + JSON validation
    ollama_client.py     # Streaming with token tracking
    conversation.py      # Message history + context management
    display.py           # Streaming buffer (hides tool_call from user)
    compaction.py        # Auto-compaction (Claude Code style LLM summarization)
    tool_context.py      # Shared state: CWD, read files, tool registry
    tools/
      __init__.py        # Registry, dispatch, output truncation
      base.py            # BaseTool, ToolResult
      bash.py            # subprocess, CWD via pwd, interactive detect
      read_file.py       # Line numbers, offset/limit, binary detect
      edit_file.py       # Exact replacement, read-first check
      write_file.py      # Create/overwrite, auto mkdir
      glob_tool.py       # os.walk based, .git pruning
      grep_tool.py       # rg -> grep -> python fallback
      list_dir.py        # Dirs first, sizes
  tests/
    test_parser.py       # Offline unit tests
```

### Shared ToolContext (NEW -- fixes state sharing problem)

All tools access shared state through a single ToolContext object:

```python
# shadow_code/tool_context.py
class ToolContext:
    def __init__(self, initial_cwd: str):
        self.cwd = initial_cwd           # updated by bash after every command
        self.read_files: set[str] = set() # tracks files read by read_file

    def mark_file_read(self, path: str):
        self.read_files.add(os.path.abspath(path))

    def was_file_read(self, path: str) -> bool:
        return os.path.abspath(path) in self.read_files
```

Every tool receives `ToolContext` in its constructor. bash.py updates `ctx.cwd` after every command. edit_file checks `ctx.was_file_read()`. glob/grep use `ctx.cwd` as default path.

---

## 3. Phase 0: Empirical Validation (BEFORE any code)

### Test 0.1: Format comparison
Test 3 formats with real model (5 runs each):
- A: `<tool_call>{"tool":...}</tool_call>`
- B: ` ```tool_call ... ``` `
- C: `>>>TOOL ... TOOL<<<`

Method: curl to Ollama API with short prompt + format instructions.
Pick format with highest accuracy (5/5).

### Test 0.2: Multi-turn
3-turn conversation: user -> tool_call -> tool_result -> answer -> user -> tool_call again.
Does model maintain format on turn 2+?

### Test 0.3: Few-shot vs Zero-shot
Same prompt, 10 tests each. Measure format compliance.

### Test 0.4: Temperature
Test 0.1, 0.3, 0.5, 0.7 (5 runs each). Pick most reliable.

### Test 0.5: Prompt length
Short (~2K chars) vs full (~6K chars). Quality same?

Output: VALIDATION_RESULTS.md with chosen format, temperature, style.

---

## 4. Phase 1: Minimal Working Version

### 4.1 pyproject.toml

```toml
[project]
name = "shadow-code"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["requests>=2.31.0"]

[project.optional-dependencies]
full = ["rich>=13.0.0", "prompt_toolkit>=3.0.0"]

[project.scripts]
shadow-code = "shadow_code.main:main"
```

### 4.2 config.py

```python
import os

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.environ.get("SHADOW_MODEL", "shadow-gemma:latest")
CONTEXT_WINDOW = 131_072
# Thresholds are in conversation.py (0.55, 0.65, 0.85)
MAX_TOOL_TURNS = 20
MAX_CONSECUTIVE_ERRORS = 3
TOOL_OUTPUT_MAX_CHARS = 30_000
BASH_DEFAULT_TIMEOUT = 120
BASH_MAX_TIMEOUT = 600
MAX_LINES_TO_READ = 2000
INTERACTIVE_CMDS = {"vim","vi","nano","less","more","top","htop","man"}
BLOCKED_PATHS = {"/dev/zero","/dev/urandom","/dev/random","/dev/stdin","/dev/stdout","/dev/stderr"}
MODEL_OPTIONS = {"temperature": 0.3, "num_ctx": CONTEXT_WINDOW}
```

### 4.3 prompt.py -- THE HEART OF THE PROJECT

Every word directly affects model behavior. **100% STATIC** -- no f-strings, no dynamic content. This maximizes Ollama KV cache hits (tested: 13x speedup).

Dynamic info (CWD, date, shell) is injected as the first user message at session start (see main.py).

```python
# NOTE: This is a regular string, NOT an f-string. No variables.
# This ensures the prompt is byte-identical across requests -> KV cache hit.

SYSTEM_PROMPT = """You are Shadow, a local AI coding assistant. You help users with software engineering tasks using the tools below.

IMPORTANT: Assist with authorized security testing, defensive security, CTF challenges, and educational contexts. Refuse requests for destructive techniques, DoS attacks, mass targeting, supply chain compromise, or detection evasion for malicious purposes.

# How to Use Tools

When you need to perform an action, write a tool call in this exact format:

<tool_call>
{"tool": "tool_name", "params": {"key": "value"}}
</tool_call>

Rules:
- JSON must have exactly two keys: "tool" (string) and "params" (object)
- You may write text before and after tool calls
- You may make multiple tool calls in one response
- NEVER write <tool_call> tags in explanations. Only use them to actually invoke a tool.

After each tool call, you will receive:

<tool_result tool="tool_name" success="true">
...output...
</tool_result>

Use the result to answer the user or continue working.

## Full Example

NOTE: The exact few-shot format (with or without JSON tool_call syntax inline)
will be determined by Phase 0 validation results. If the model follows JSON
examples reliably, we include them. If text descriptions work better, we use those.
Phase 0 Test 0.3 (few-shot vs zero-shot) decides this.

Placeholder -- one of these two styles will be chosen:

Style A (JSON -- if Phase 0 confirms model follows it):
User: "what files are here?"
Assistant: Let me check.
<tool_call> {"tool":"bash","params":{"command":"ls -la"}} </tool_call>
(receives tool_result, then responds with file listing)

Style B (text description -- fallback if JSON examples confuse the model):
User: "what files are here?"
Assistant calls bash with "ls -la", gets listing, responds with summary.

# Available Tools

## bash
Executes a bash command and returns stdout/stderr.
Parameters: command (string, required), timeout (integer, optional, default 120, max 600).
Prefer dedicated tools: read_file over cat, edit_file over sed, write_file over echo, glob over find, grep over grep command.
For git: prefer new commits over amending, never skip hooks, never force push unless asked.

## read_file
Reads a file with line numbers (cat -n format).
Parameters: file_path (absolute, required), offset (1-based, optional), limit (default 2000, optional).
Always use absolute paths.

## edit_file
Exact string replacement in a file. MUST read file first with read_file.
Parameters: file_path (required), old_string (required), new_string (required), replace_all (optional, default false).
FAILS if old_string is not unique (unless replace_all=true). Preserve exact indentation.

## write_file
Creates or overwrites a file. For existing files prefer edit_file.
Parameters: file_path (absolute, required), content (required).

## glob
Finds files matching a pattern. Returns paths sorted by modification time, max 100.
Parameters: pattern (required, like "**/*.py"), path (optional, default working directory).

## grep
Searches file contents with regex. Returns file:line:content.
Parameters: pattern (required), path (optional), include (optional, like "*.py"), case_insensitive (optional), max_results (optional, default 200).

## list_dir
Lists directory contents with sizes. Dirs first, then files.
Parameters: path (optional, default working directory).

# Doing Tasks
- Read code before modifying it. Do not propose changes to unread code.
- Do not create files unless necessary. Prefer editing existing files.
- Do not add features beyond what was asked.
- Do not add error handling for impossible scenarios.
- If an approach fails, diagnose why before switching tactics.
- Be careful not to introduce security vulnerabilities.

# Executing Actions with Care
Consider reversibility. For destructive or hard-to-reverse actions, check with the user first.

# Tone and Style
- No emojis unless requested. Be concise. Lead with action, not reasoning.
- Reference code as file_path:line_number.

# Language
Respond in the same language the user writes in. You understand Georgian and English.
Technical terms and code remain in original form.
\"\"\"
```

**Dynamic info injected at session start (NOT in system prompt):**
```python
# main.py -- first user message includes environment
ENV_PREFIX = f"[Environment: CWD={cwd}, Platform={plat}, Shell={shell}, Date={date}]"
# Prepended to the very first user message only
```

Prompt stats: ~3,500 chars, ~1,200 Gemma tokens, 100% static (KV cache friendly), 2 few-shot examples, 7 tool descriptions.

### 4.4 parser.py

```python
import re, json
from dataclasses import dataclass

@dataclass
class ToolCall:
    tool: str
    params: dict
    raw: str

TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL)

def parse_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Extract tool calls. Returns (clean_text, calls)."""
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data.get("tool"), str) and isinstance(data.get("params"), dict):
                calls.append(ToolCall(tool=data["tool"], params=data["params"], raw=match.group(0)))
            # else: not a real tool call, skip
        except json.JSONDecodeError:
            calls.append(ToolCall(tool="__invalid__",
                params={"error": f"Invalid JSON: {match.group(1)[:200]}"},
                raw=match.group(0)))
    clean = TOOL_CALL_RE.sub("", text).strip()
    return clean, calls
```

### 4.5 ollama_client.py

```python
import requests, json
from .config import OLLAMA_BASE_URL, MODEL_NAME, MODEL_OPTIONS

class OllamaClient:
    def __init__(self):
        self.last_prompt_tokens = 0
        self.last_eval_tokens = 0

    def health_check(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            if MODEL_NAME not in models:
                return False, f"Model '{MODEL_NAME}' not found. Available: {models}"
            return True, "OK"
        except requests.ConnectionError:
            return False, f"Cannot connect to Ollama at {OLLAMA_BASE_URL}"

    def chat_stream(self, messages: list[dict], system: str):
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": True,
            "options": MODEL_OPTIONS,
        }
        resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat",
            json=payload, stream=True, timeout=600)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line: continue
            data = json.loads(line)
            content = data.get("message", {}).get("content", "")
            if content:
                yield content
            if data.get("done"):
                self.last_prompt_tokens = data.get("prompt_eval_count", 0)
                self.last_eval_tokens = data.get("eval_count", 0)
```

### 4.6 conversation.py

```python
from .config import CONTEXT_WINDOW

# Thresholds (Claude Code style)
RESULT_CLEARING_RATIO = 0.55   # clear old tool results at 55%
COMPACTION_RATIO = 0.65        # auto-compact at 65% (low enough to leave room for summary)
EMERGENCY_TRUNCATE_RATIO = 0.85 # hard truncate at 85% (fallback if compaction fails)
KEEP_RECENT_RESULTS = 5
KEEP_RECENT_MESSAGES = 20
CLEARED_STUB = "[Tool result cleared to save context. Output was already processed.]"

class Conversation:
    def __init__(self):
        self.messages: list[dict] = []
        self.total_prompt_tokens = 0

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, results_text: str):
        prefixed = "[Tool execution results. Continue working on the user's request.]\n\n" + results_text
        self.messages.append({"role": "user", "content": prefixed})

    def update_tokens(self, prompt_tokens: int):
        self.total_prompt_tokens = prompt_tokens

    # === Context Management (Claude Code style, 3-tier) ===

    def needs_result_clearing(self) -> bool:
        return self.total_prompt_tokens > int(CONTEXT_WINDOW * RESULT_CLEARING_RATIO)

    def needs_compaction(self) -> bool:
        return self.total_prompt_tokens > int(CONTEXT_WINDOW * COMPACTION_RATIO)

    def needs_emergency_truncate(self) -> bool:
        return self.total_prompt_tokens > int(CONTEXT_WINDOW * EMERGENCY_TRUNCATE_RATIO)

    def clear_old_tool_results(self):
        """Tier 1: Replace old tool results with stubs. Fast, no API call."""
        indices = [i for i, m in enumerate(self.messages)
                   if m["role"] == "user" and "<tool_result" in m["content"]]
        for i in indices[:-KEEP_RECENT_RESULTS]:
            self.messages[i] = {"role": "user", "content": CLEARED_STUB}

    def apply_compaction_summary(self, summary: str):
        """Tier 2: Replace old messages with LLM-generated summary."""
        if len(self.messages) <= KEEP_RECENT_MESSAGES:
            return
        self.messages = self.messages[-KEEP_RECENT_MESSAGES:]
        self.messages.insert(0, {"role": "user",
            "content": "[This session continues from earlier. Summary of previous work:\n\n"
                       + summary + "\n\nRecent messages follow.]"})

    def emergency_truncate(self):
        """Tier 3: Hard truncate as last resort (compaction failed)."""
        if len(self.messages) <= KEEP_RECENT_MESSAGES:
            return
        removed = len(self.messages) - KEEP_RECENT_MESSAGES
        self.messages = self.messages[-KEEP_RECENT_MESSAGES:]
        self.messages.insert(0, {"role": "user",
            "content": f"[Previous {removed} messages truncated to save context.]"})

    def clear(self):
        self.messages.clear()
        self.total_prompt_tokens = 0

    def get_messages(self) -> list[dict]:
        return list(self.messages)
```

### 4.7 display.py -- Streaming Buffer

Solves C3: user never sees raw `<tool_call>` JSON.

```python
import sys

TAG_START = "<tool_call>"
TAG_END = "</tool_call>"

class StreamDisplay:
    def __init__(self):
        self.buffer = ""
        self.buffering = False
        self.full_response = ""

    def reset(self):
        self.buffer = ""
        self.buffering = False
        self.full_response = ""

    def feed(self, chunk: str):
        self.full_response += chunk
        if self.buffering:
            self.buffer += chunk
            if TAG_END in self.buffer:
                self.buffering = False
                after = self.buffer.split(TAG_END, 1)[-1]
                if after.strip():
                    sys.stdout.write(after)
                    sys.stdout.flush()
                self.buffer = ""
            return

        combined = self.buffer + chunk
        if TAG_START in combined:
            before = combined.split(TAG_START, 1)[0]
            after = combined.split(TAG_START, 1)[1]
            if before.strip():
                sys.stdout.write(before)
                sys.stdout.flush()
            self.buffering = True
            self.buffer = TAG_START + after
            return

        # Check partial match at boundary
        for i in range(min(len(TAG_START), len(combined)), 0, -1):
            if TAG_START.startswith(combined[-i:]):
                safe = combined[:-i]
                if safe:
                    sys.stdout.write(safe)
                    sys.stdout.flush()
                self.buffer = combined[-i:]
                return

        sys.stdout.write(combined)
        sys.stdout.flush()
        self.buffer = ""

    def flush(self):
        if self.buffer:
            sys.stdout.write(self.buffer)
            sys.stdout.flush()
            self.buffer = ""

    def get_full_response(self) -> str:
        return self.full_response
```

### 4.8 tools/base.py

```python
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ToolResult:
    success: bool
    output: str

class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def execute(self, params: dict) -> ToolResult: ...

    def validate(self, params: dict) -> str | None:
        return None
```

### 4.9 tools/__init__.py

```python
from .base import BaseTool, ToolResult
from ..config import TOOL_OUTPUT_MAX_CHARS

_REGISTRY: dict[str, BaseTool] = {}

def register(tool: BaseTool):
    _REGISTRY[tool.name] = tool

def dispatch(name: str, params: dict) -> ToolResult:
    tool = _REGISTRY.get(name)
    if not tool:
        avail = ", ".join(_REGISTRY.keys())
        return ToolResult(False, f"Unknown tool: '{name}'. Available: {avail}")
    err = tool.validate(params)
    if err:
        return ToolResult(False, f"Invalid parameters: {err}")
    try:
        result = tool.execute(params)
        result.output = _truncate(result.output)
        return result
    except Exception as e:
        return ToolResult(False, f"Tool error: {type(e).__name__}: {e}")

def _truncate(text: str) -> str:
    if len(text) <= TOOL_OUTPUT_MAX_CHARS: return text
    half = TOOL_OUTPUT_MAX_CHARS // 2
    removed = len(text) - TOOL_OUTPUT_MAX_CHARS
    return text[:half] + f"\n\n[...{removed} chars truncated...]\n\n" + text[-half:]

def format_result(tool_name: str, result: ToolResult) -> str:
    s = "true" if result.success else "false"
    return f'<tool_result tool="{tool_name}" success="{s}">\n{result.output}\n</tool_result>'
```

### 4.10 tools/bash.py

```python
import subprocess, os, signal, re
from .base import BaseTool, ToolResult
from ..config import BASH_DEFAULT_TIMEOUT, BASH_MAX_TIMEOUT, INTERACTIVE_CMDS

class BashTool(BaseTool):
    name = "bash"

    def __init__(self, ctx):  # ctx: ToolContext
        self.ctx = ctx

    def validate(self, params: dict) -> str | None:
        if "command" not in params: return "Missing required: command"
        cmd = params["command"].strip()
        if not cmd: return "Command cannot be empty"
        first = cmd.split()[0].split("/")[-1]
        if first in INTERACTIVE_CMDS:
            return f"Interactive command '{first}' not supported"
        return None

    def execute(self, params: dict) -> ToolResult:
        command = params["command"]
        timeout = min(params.get("timeout", BASH_DEFAULT_TIMEOUT), BASH_MAX_TIMEOUT)
        try:
            proc = subprocess.Popen(command, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, text=True,
                cwd=self.ctx.cwd, preexec_fn=os.setsid)
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            return ToolResult(False, f"Timed out after {timeout}s")
        except Exception as e:
            return ToolResult(False, f"Error: {e}")

        # CWD tracking: detect cd/pushd commands only (SAFE -- never re-executes command)
        if proc.returncode == 0:
            import shlex
            try:
                tokens = shlex.split(command)
            except ValueError:
                tokens = command.split()
            if tokens and tokens[0] in ("cd", "pushd"):
                target = tokens[1] if len(tokens) > 1 else os.path.expanduser("~")
                new = os.path.normpath(os.path.join(self.ctx.cwd, os.path.expanduser(target)))
                if os.path.isdir(new):
                    self.ctx.cwd = new

        parts = []
        if stdout.strip(): parts.append(stdout.rstrip())
        if stderr.strip(): parts.append(f"STDERR: {stderr.rstrip()}")
        if proc.returncode != 0: parts.append(f"(exit code: {proc.returncode})")
        return ToolResult(proc.returncode == 0, "\n".join(parts) or "(no output)")
```

### 4.11 main.py

```python
import sys, os, signal, platform
from datetime import datetime
from .config import MODEL_NAME, MAX_TOOL_TURNS, MAX_CONSECUTIVE_ERRORS, CONTEXT_WINDOW
from .ollama_client import OllamaClient
from .prompt import SYSTEM_PROMPT  # static constant, NOT a function
from .parser import parse_tool_calls
from .conversation import Conversation
from .display import StreamDisplay
from .tool_context import ToolContext
from . import tools as tool_reg

def main():
    cwd = os.getcwd()
    ctx = ToolContext(cwd)  # shared state for all tools

    # Register tools (Phase 1: only bash)
    from .tools.bash import BashTool
    tool_reg.register(BashTool(ctx))

    client = OllamaClient()
    ok, msg = client.health_check()
    if not ok:
        print(f"Error: {msg}")
        sys.exit(1)

    print(f"shadow-code v0.1.0 | {MODEL_NAME}")
    print(f"CWD: {cwd}")
    print("Commands: /help /clear /exit /tokens\n")

    conv = Conversation()
    display = StreamDisplay()
    interrupted = False
    first_message = True  # track first message for env prefix

    def on_sigint(s, f):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, on_sigint)

    while True:
        try:
            user_input = input("shadow> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break

        if not user_input: continue
        if user_input == "/exit": break
        if user_input == "/clear":
            conv.clear(); first_message = True; print("[Cleared]"); continue
        if user_input == "/tokens":
            pct = (conv.total_prompt_tokens / CONTEXT_WINDOW * 100) if CONTEXT_WINDOW else 0
            print(f"Prompt: {conv.total_prompt_tokens} tokens ({pct:.0f}% of {CONTEXT_WINDOW})"); continue
        if user_input == "/help":
            print("/clear /exit /tokens"); continue

        # Inject environment info on first message (keeps system prompt static for KV cache)
        if first_message:
            shell = os.environ.get("SHELL", "/bin/bash").rsplit("/", 1)[-1]
            env_prefix = (f"[Environment: CWD={ctx.cwd}, "
                         f"Platform={platform.system()} {platform.release()}, "
                         f"Shell={shell}, Date={datetime.now().strftime('%Y-%m-%d')}]\n\n")
            conv.add_user(env_prefix + user_input)
            first_message = False
        else:
            conv.add_user(user_input)
        turns = 0
        errors = 0

        while turns < MAX_TOOL_TURNS:
            interrupted = False
            display.reset()
            try:
                for chunk in client.chat_stream(conv.get_messages(), SYSTEM_PROMPT):
                    if interrupted: break
                    display.feed(chunk)
            except KeyboardInterrupt:
                interrupted = True
            except Exception as e:
                print(f"\n[Error: {e}]"); break

            display.flush()
            print()

            resp = display.get_full_response()
            if not resp.strip(): break
            conv.add_assistant(resp)
            conv.update_tokens(client.last_prompt_tokens)

            if interrupted:
                print("[Interrupted]"); break

            _, calls = parse_tool_calls(resp)
            if not calls: break

            results = []
            for tc in calls:
                if tc.tool == "__invalid__":
                    r = tool_reg.ToolResult(False, tc.params.get("error","Invalid"))
                    print(f"  [error] {r.output}")
                else:
                    desc = tc.params.get("command", tc.params.get("file_path", str(tc.params)))[:80]
                    print(f"  [{tc.tool}] {desc}")
                    r = tool_reg.dispatch(tc.tool, tc.params)
                    preview = r.output[:200] + ("..." if len(r.output) > 200 else "")
                    for line in preview.split("\n")[:5]:
                        print(f"    {line}")
                results.append(tool_reg.format_result(tc.tool, r))
                errors = errors + 1 if not r.success else 0

            conv.add_tool_results("\n\n".join(results))
            turns += 1
            if errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"[{errors} consecutive errors]"); break

        if turns >= MAX_TOOL_TURNS:
            print(f"[Tool limit ({MAX_TOOL_TURNS}) reached]")

        # === 3-Tier Context Management (Claude Code style) ===
        conv.update_tokens(client.last_prompt_tokens)

        # Tier 1: Clear old tool results (fast, no API call)
        if conv.needs_result_clearing():
            conv.clear_old_tool_results()

        # Tier 2: Auto-compaction (LLM summarization)
        if conv.needs_compaction():
            print("[Compacting conversation...]")
            try:
                from .compaction import compact
                summary = compact(client, conv.get_messages(), SYSTEM_PROMPT)
                conv.apply_compaction_summary(summary)
                print("[Compaction complete]")
            except Exception as e:
                print(f"[Compaction failed: {e}]")
                # Tier 3: Emergency truncate (fallback)
                if conv.needs_emergency_truncate():
                    conv.emergency_truncate()
                    print("[Emergency truncation applied]")

if __name__ == "__main__":
    main()
```

---

## 5. Phase 2: Full Tool Set

### 5.1 read_file.py
- Line numbers cat -n format (6-char pad + tab)
- offset (1-based), limit (default 2000)
- Binary detection (null bytes in first 8KB)
- Encoding: utf-8, fallback latin-1
- Blocked paths (/dev/*)
- Directory check -> error
- Track read files (shared set) for edit_file validation

### 5.2 write_file.py
- Absolute path validation (must start with /)
- Auto-create parent dirs (os.makedirs)
- Return "Created new file" / "Updated file (X -> Y bytes)"

### 5.3 edit_file.py
- Uniqueness check: if replace_all=false and >1 match -> error
- old_string not found -> error
- old_string == new_string -> error
- Verify file was previously read (check shared read set)
- Return "Replaced N occurrence(s)"

### 5.4 glob_tool.py
- os.walk based (NOT pathlib.glob) for .git pruning during walk
- Prune: .git, node_modules, __pycache__, .venv, .mypy_cache
- Max 100 results
- Sort by mtime (newest first)

### 5.5 grep_tool.py
3-tier fallback:
1. `rg` (ripgrep) subprocess -- fastest
2. `grep -rn` (system) subprocess -- everywhere
3. Python `re` + `os.walk` -- guaranteed

- Auto-exclude .git, node_modules
- Binary skip
- Output: file:line:content
- max_results default 200

### 5.6 list_dir.py
- Dirs first (with / suffix), then files
- Human-readable sizes (KB, MB)
- Alphabetical within groups
- Hidden files included

---

## 6. Phase 2.5: Claude Code-style Context Management (NEW)

Claude Code-ის ზუსტი მექანიზმების იმპლემენტაცია.

### 6.0 KV Cache Optimization

Ollama-ს KV cache ტესტით დადასტურდა:
- Request 1: 3596ms (cold)
- Request 2: 268ms (cached, 13x faster!)

**Strategy:** system prompt უნდა იყოს 100% სტატიკური (არასოდეს შეიცვლება სესიის განმავლობაში). დინამიკური ინფორმაცია (CWD, date) პირველ user message-ში იგზავნება.

```python
# prompt.py -- SYSTEM_PROMPT is a module-level constant (not a function)
from .prompt import SYSTEM_PROMPT  # byte-identical every request -> KV cache hit

# main.py -- dynamic info as first user message (see section 4.11)
# first_message flag ensures env prefix is added only once
```

### 6.1 Auto-Compaction (Claude Code-ის მსგავსი)

Context window-ის **65%**-ზე ტრიგერდება (არა 75% -- საკმარისი ადგილი უნდა დარჩეს summary-სთვის). **ცალკე Ollama API call**-ით LLM აჯამებს საუბარს.

**Compaction prompt (Claude Code-დან ადაპტირებული):**

```python
COMPACT_PROMPT = """Your task is to create a detailed summary of the conversation so far.
This summary will replace the conversation history to free up context space.

CRITICAL: Respond with TEXT ONLY. Do NOT use any tools.

Your summary should include:

1. Primary Request: What the user asked for
2. Files Modified: List of files read, edited, or created with key changes
3. Errors and Fixes: Problems encountered and how they were resolved
4. Current Work: What was being worked on immediately before this summary
5. Pending Tasks: Any incomplete work
6. Next Step: The next action to take

Wrap your analysis in <analysis> tags (will be stripped), then provide the
summary in <summary> tags.

<analysis>
[Your analysis of the conversation]
</analysis>

<summary>
[Structured summary following the sections above]
</summary>
"""
```

**Implementation (`shadow_code/compaction.py`):**

```python
class Compactor:
    def __init__(self, client: OllamaClient):
        self.client = client

    def should_compact(self, prompt_tokens: int, threshold: int) -> bool:
        return prompt_tokens > threshold

    def compact(self, messages: list[dict], system: str) -> str:
        """Send conversation to LLM for summarization. Returns summary text."""
        compact_messages = messages + [{"role": "user", "content": COMPACT_PROMPT}]
        full_response = ""
        for chunk in self.client.chat_stream(compact_messages, system):
            full_response += chunk
        return format_summary(full_response)

    def format_summary(self, raw: str) -> str:
        """Strip <analysis>, extract <summary>."""
        # Remove analysis block
        import re
        cleaned = re.sub(r'<analysis>.*?</analysis>', '', raw, flags=re.DOTALL)
        # Extract summary
        match = re.search(r'<summary>(.*?)</summary>', cleaned, re.DOTALL)
        if match:
            return match.group(1).strip()
        return cleaned.strip()
```

**Conversation after compaction:**
```python
messages = [
    {"role": "user", "content": "[This session continues from a previous conversation. Summary:\n\n" + summary + "\n\nRecent messages follow.]"},
    # ... last few messages preserved ...
]
```

### 6.2 Function Result Clearing (Claude Code-ის მსგავსი)

ძველი tool result-ების შეკვეცა. ბოლო 5 tool result სრულად რჩება, ძველები იკვეცება.

```python
KEEP_RECENT_TOOL_RESULTS = 5
CLEARED_STUB = "[Tool result cleared to save context space. The output was already processed.]"

def clear_old_tool_results(messages: list[dict]) -> list[dict]:
    """Replace old tool results with stubs, keep last N intact."""
    tool_result_indices = []
    for i, msg in enumerate(messages):
        if msg["role"] == "user" and "<tool_result" in msg["content"]:
            tool_result_indices.append(i)

    # Keep last KEEP_RECENT_TOOL_RESULTS, clear the rest
    to_clear = tool_result_indices[:-KEEP_RECENT_TOOL_RESULTS]
    for i in to_clear:
        messages[i] = {"role": "user", "content": CLEARED_STUB}

    return messages
```

### 6.3 Context Management Flow (3-Tier, Claude Code style)

```
ყოველ turn-ის ბოლოს:
  1. Token count update (Ollama prompt_eval_count)

  2. Tier 1: If > 55% context used:
     -> Clear old tool results (stub-ით ჩანაცვლება)
     -> Fast, no API call, saves ~10-20%

  3. Tier 2: If > 65% context used:
     -> Auto-compact (LLM summarization via separate API call)
     -> Summary replaces old messages, keeps last 20
     -> Saves ~40-60%

  4. Tier 3: If > 85% AND compaction failed:
     -> Emergency truncate (hard delete old messages)
     -> Last resort, no API call

  5. KV cache automatically handles prefix caching
```

---

## 7. Phase 3: Polish (deepseek-chat UI patterns reused)

Source: github.com/null0xxx/deepseek-chat -- same author's Python CLI with Rich UI.
We reuse its proven UI architecture but add tool calling display.

### 7.1 ui.py -- UIRenderer (from deepseek-chat)

```python
from rich.console import Group
from rich.panel import Panel
from rich.markdown import Markdown
from rich.text import Text
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table

class UIRenderer:
    def render_welcome(self) -> Panel:
        content = Text()
        content.append("Shadow Code", style="bold cyan")
        content.append(f" v0.1.0\n", style="dim")
        content.append("Local AI Coding Assistant\n", style="dim")
        content.append("Model: shadow-gemma:latest | ", style="dim")
        content.append("Type /help for commands", style="dim italic")
        return Panel(content, border_style="cyan", padding=(1, 2))

    def render_thinking(self, model: str) -> Panel:
        spinner = Spinner("dots", text=Text(f" {model} is thinking...", style="cyan"))
        return Panel(spinner, border_style="blue")

    def render_streaming(self, text: str) -> Panel:
        return Panel(
            Markdown(text) if text.strip() else Text("..."),
            border_style="blue",
            subtitle="[dim]streaming...[/dim]",
        )

    def render_response(self, text: str, tokens: int = 0) -> Panel:
        subtitle = f"[dim]{tokens:,} tokens[/dim]" if tokens else ""
        return Panel(Markdown(text), border_style="blue", subtitle=subtitle)

    def render_tool_call(self, tool: str, desc: str) -> Panel:
        return Panel(
            Text(desc, style="dim"),
            border_style="cyan",
            title=f"[bold cyan]{tool}[/bold cyan]",
        )

    def render_tool_result(self, tool: str, output: str, success: bool) -> Panel:
        style = "green" if success else "red"
        preview = output[:500] + ("..." if len(output) > 500 else "")
        return Panel(
            Text(preview),
            border_style=style,
            title=f"[bold {style}]{tool}[/bold {style}]",
        )

    def render_error(self, message: str) -> Panel:
        return Panel(Text(message, style="bold red"), border_style="red",
                     title="[bold red]Error[/bold red]")

    def render_context_status(self, used: int, total: int) -> Text:
        pct = (used / total) * 100 if total else 0
        color = "green" if pct < 50 else "yellow" if pct < 75 else "red"
        text = Text()
        text.append(f"  Context: ", style="dim")
        text.append(f"{used//1000}K/{total//1000}K", style=f"bold {color}")
        text.append(f" ({pct:.0f}%)", style="dim")
        return text
```

### 7.2 streaming.py -- StreamController with Rich Live (from deepseek-chat)

```python
from rich.live import Live
from rich.console import Console

class StreamController:
    def __init__(self, client, ui, console):
        self.client = client
        self.ui = ui
        self.console = console

    def stream_response(self, messages, system) -> tuple[str, int]:
        """Stream with live Rich panel. Returns (full_text, eval_tokens)."""
        accumulated = ""
        buffer = StreamBuffer()  # our tool_call buffer logic

        with Live(self.ui.render_thinking("shadow-gemma"),
                  console=self.console, refresh_per_second=10) as live:
            for chunk in self.client.chat_stream(messages, system):
                text = buffer.feed(chunk)
                if text:
                    accumulated += text
                    live.update(self.ui.render_streaming(accumulated))

            buffer.flush()
            live.update(self.ui.render_response(
                accumulated, self.client.last_eval_tokens))

        return buffer.get_full_response(), self.client.last_eval_tokens
```

Key difference from deepseek-chat: our StreamBuffer hides `<tool_call>` JSON.
The Rich Live panel shows clean markdown, never raw JSON.

### 7.3 prompt.py REPL (from deepseek-chat)

```python
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import WordCompleter

def create_prompt_session() -> PromptSession:
    history_path = Path("~/.shadow-code/prompt_history").expanduser()
    history_path.parent.mkdir(parents=True, exist_ok=True)

    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def multiline(event):
        event.current_buffer.insert_text("\n")

    @bindings.add("c-d")
    def exit_handler(event):
        event.app.exit(result=None)

    completer = WordCompleter(["/help","/clear","/exit","/tokens",
                               "/save","/load","/model","/info"])

    return PromptSession(
        history=FileHistory(str(history_path)),
        key_bindings=bindings,
        completer=completer,
        multiline=False,
        enable_history_search=True,
    )
```

### 7.4 db.py -- Session Persistence (from deepseek-chat)

SQLite with WAL mode, same pattern as deepseek-chat:
- ~/.shadow-code/sessions.db
- Tables: sessions (id, model, created, updated), messages (session_id, role, content, timestamp)
- /save, /load, /list, /delete commands
- Auto-save on clean exit

### 7.5 Destructive command warnings

```python
DESTRUCTIVE = [r'\brm\s+-rf\b', r'\bgit\s+push\s+--force\b',
               r'\bgit\s+reset\s+--hard\b', r'\bdrop\s+table\b']
```
User confirmation via prompt_toolkit before execution.

### 7.6 Context status

After each turn, show context usage (color-coded):
```
  Context: 45K/128K (35%)     <- green
  Context: 85K/128K (66%)     <- yellow
  Context: 110K/128K (86%)    <- red
```

---

## 8. Streaming Buffer Design

```
Stream: "Let me check." -> print immediately
Stream: "<tool_"        -> hold in buffer (might be <tool_call>)
Stream: "_call>{"       -> confirmed, keep buffering
Stream: "...}</tool_"   -> keep buffering
Stream: "_call>"        -> complete! parse & execute, show [bash] ls -la

If "<tool_" + "s are great" -> not a tag, flush "tools are great"
```

User sees: text + [tool_name] + tool output + more text
User NEVER sees: raw <tool_call> JSON

---

## 9. Tool Result Format

Sent to model as "user" role with context prefix:

```
[Tool execution results. Continue working on the user's request.]

<tool_result tool="bash" success="true">
total 16
-rw-r--r-- 1 user user 500 main.py
</tool_result>
```

Prefix prevents model from treating tool results as new user requests.

---

## 10. Error Handling

| Scenario | Detection | Resolution |
|----------|-----------|------------|
| Ollama not running | health_check() | Error message + exit |
| Model not found | health_check() | "Not found. Available: [list]" |
| Invalid JSON in tool_call | JSONDecodeError | Error in tool_result, model self-corrects |
| Unknown tool name | Registry miss | "Unknown. Available: [list]" |
| Tool crash | try/except | Error in tool_result |
| Bash timeout | TimeoutExpired | Kill process group, error |
| Interactive command | validate() | "Not supported" |
| File not found | FileNotFoundError | Error in tool_result |
| Tool loop (20+) | Counter | "[Limit reached]" + break |
| Consecutive errors (3+) | Counter | Warning + break |
| Large output (30K+) | _truncate() | head + "...truncated..." + tail |
| Context overflow | prompt_eval_count | Truncate old messages |
| Network error | requests exception | Print error, return to prompt |
| Ctrl+C during stream | Signal handler | "[Interrupted]", return to prompt |

---

## 11. Verification Tests

| # | Test | Expected |
|---|------|----------|
| 1 | Start (Ollama running) | Header + prompt |
| 2 | Start (Ollama down) | Error + exit |
| 3 | "gamarjoba" | Georgian response, no tools |
| 4 | "list files here" | bash -> ls output |
| 5 | "read main.py" | read_file -> line numbers |
| 6 | "create test.txt with hello" | write_file -> created |
| 7 | "replace hello with world in test.txt" | edit_file -> replaced |
| 8 | "find all .py files" | glob -> file list |
| 9 | "search for import in .py" | grep -> matches |
| 10 | Read nonexistent file | Error, model recovers |
| 11 | Ctrl+C during stream | "[Interrupted]" |
| 12 | /clear | "[Cleared]" |
| 13 | Multi-tool chain | read -> analyze -> edit |
| 14 | tool_call not visible to user | Buffer works |
| 15 | 20+ tool calls | Limit message |
| 16 | Long conversation (50+ turns) | Auto-compaction triggers |
| 17 | Compaction failure | Emergency truncate fallback |
| 18 | CWD change via cd | Next bash uses new CWD |

---

## 12. Implementation Order

```
Phase 0 (Validation):
  0a. Format testing (curl, 3 formats x 5 runs)
  0b. Multi-turn test
  0c. Temperature optimization
  0d. VALIDATION_RESULTS.md

Phase 1 (Minimal):
  1.  pyproject.toml + __init__.py
  2.  config.py
  3.  tool_context.py (shared state: CWD, read files)
  4.  tools/base.py + tools/__init__.py
  5.  tools/bash.py (CWD tracking via pwd)
  6.  ollama_client.py
  7.  parser.py
  8.  conversation.py (with 3-tier context management)
  9.  display.py (streaming buffer)
  10. prompt.py (100% STATIC, no dynamic content)
  11. main.py (env info as first user message)
  12. tests/test_parser.py
  TEST: "list files" end-to-end

Phase 2 (Full tools):
  13. tools/read_file.py
  14. tools/write_file.py
  15. tools/edit_file.py
  16. tools/glob_tool.py
  17. tools/grep_tool.py
  18. tools/list_dir.py
  TEST: all tools end-to-end

Phase 2.5 (Context Management -- Claude Code style):
  19. KV cache optimization (static system prompt, dynamic in user msg)
  20. shadow_code/compaction.py (auto-compaction via LLM summarization)
  21. Function result clearing (old tool results -> stubs)
  22. Context management flow in main.py
  TEST: long conversation triggers compaction

Phase 3 (Polish -- deepseek-chat UI patterns):
  23. ui.py (Rich panels, markdown, spinners)
  24. streaming.py (Rich Live + StreamBuffer)
  25. prompt.py REPL (prompt_toolkit + FileHistory + completer)
  26. db.py (SQLite session persistence)
  27. Destructive command warnings
  28. Context status display
  FULL TESTING
```

---

## 13. Out of Scope

- Vision/Image support
- PDF reading
- MCP server integration
- Agent/Subagent system
- Permission system (local, user=owner)
- Anthropic-style server-side prompt caching (Ollama KV cache IS used)
- Modelfile modification
- Web UI
- vecmem/RAG integration (analyzed, decided against -- Claude Code style caching chosen instead)
