# shadow-code × qwen2.5-coder × EAS

> **Setup guide for running shadow-code with `qwen2.5-coder` (Ollama) plus
> EAS (Elite Agent System) engineering rules and skills.**

This variant pairs the local `qwen2.5-coder:7b` model with shadow-code's
agentic loop and selectively imports rules + skills from
[`elite-agent-system`](https://github.com/null0xxx/elite-agent-system) at
`~/.copilot/`.

## What you get

| Capability | How |
|---|---|
| Local function-calling agent | shadow-code's parser + tool registry |
| qwen-tuned generation | `Modelfile.qwen` (ChatML, 32K ctx, temp 0.2) |
| EAS Iron Laws + security baseline | injected into `SYSTEM_PROMPT` (~2KB) |
| On-demand language rules | `get_language_rules` tool reads `~/.copilot/rules/*.md` |
| Distilled EAS workflows | `/review`, `/test`, `/audit`, `/migrate` skills |

## Prerequisites

```bash
# 1. Ollama with qwen2.5-coder
ollama pull qwen2.5-coder:latest

# 2. EAS rules installed at ~/.copilot/rules/
# (only required if you want the get_language_rules tool to return content)
ls ~/.copilot/rules/   # should list common-security.md, python.md, ...
```

## Build the qwen-tuned model

```bash
cd shadow-code
ollama create shadow-qwen -f Modelfile.qwen
```

`Modelfile.qwen` sets:
- `FROM qwen2.5-coder:latest`
- `parameter num_ctx 32768`
- `parameter temperature 0.2`
- `parameter top_p 0.9` / `top_k 40` / `min_p 0.05`
- `parameter num_predict 8192`
- `parameter repeat_penalty 1.15` + `repeat_last_n 256` (prevents Q4
  degeneration loops on low-frequency Georgian/Cyrillic tokens)
- `parameter stop "<|im_end|>"` (qwen ChatML stop)

## Install & run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install rich prompt_toolkit  # optional UI deps

SHADOW_MODEL=shadow-qwen:latest shadow-code
```

## What's qwen-specific

| File | Change |
|---|---|
| `Modelfile.qwen` | New: qwen2.5 ChatML stop tokens, 32K ctx |
| `shadow_code/parser.py` | 4-pass extraction: canonical `tool_call` fence → lenient JSON-shape fence → unclosed-fence fallback → **raw `bash`/`sh` fence as bash tool call** (Pass 4 catches 7B's tendency to skip the JSON wrapper entirely) |
| `shadow_code/prompt.py` | Anti-hallucination block; EAS Iron Laws + **full 20 zero-trust security rules**; `get_language_rules` tool documentation |
| `shadow_code/rules_loader.py` | New: extension→rule mapping + cached summary loader from `~/.copilot/rules/` |
| `shadow_code/tools/get_language_rules.py` | New: BaseTool for on-demand rule fetch |
| `shadow_code/skills.py` | `/review` enriched with 5-Eye protocol; `/test` enriched with testing pyramid + `--e2e`; new `/audit` (STRIDE security) and `/migrate` (DB migrations) |

## Skills overview (20 total)

```
Planning & review:
  /plan            Implementation plan (no edits) — V1-V7 schema
  /review          4-section structured: ARCHITECT / SECURITY / QA / LEAD + V1-V7
  /audit           STRIDE security audit
  /cross-validate  Adversarial 2-pass self-review

Code quality:
  /harden          Production resilience: errors, edges, i18n, concurrency
  /distill         Simplification preserving behavior
  /refactor        Behavior-preserving cleanup
  /simplify        Reuse / quality / efficiency review
  /debug           Root-cause + fix

Domain workflows:
  /test            Testing pyramid; --e2e for Playwright
  /migrate         Safe DB migrations (PG/MySQL/Prisma/Drizzle/Django)
  /api-design      REST API design/audit checklist

Utilities:
  /commit /pr /verify /search /explain /init /stuck /remember
```

All skills auto-prepend a bilingual directive: respond in the user's language
(Georgian if user wrote Georgian; identifiers/paths/code always English).

## Tools

`bash`, `read_file`, `edit_file`, `write_file`, `glob`, `grep`, `list_dir`,
`multi_read`, `project_summary`, `file_backup`, `file_restore`,
**`get_language_rules`**.

`get_language_rules` accepts `{"extension": ".py"}` or `{"name": "python"}`,
returns a ~1.5KB summary. Pass `{"full": true}` for the full rule.

`edit_file` and `write_file` automatically nudge the model toward
`get_language_rules` on substantive changes (deduped per session).

## Validation

| Phase | Suite | Result |
|---|---|---|
| 0 | 15 prompt tool-format probe | 87–100% avg 3.0s (7B stochastic) |
| 5 | 5 realistic agentic tasks (read/grep/ls/rules/bash) | 5/5 (100%), avg 3.4s |

Run yourself:

```bash
python phase0_validate.py    # tool-format reliability
python phase5_e2e.py         # realistic task probes
```

## Token budget

| Component | Chars | Tokens | % of 32K |
|---|---|---|---|
| System prompt | 10,892 | ~2,723 | 8.3% |
| Largest skill (`/migrate`) | 1,759 | ~440 | 1.3% |
| **Worst-case static** | **~12,650** | **~3,162** | **9.7%** |
| Available for work | — | ~29,600 | 90.3% |

## Limitations

- 7B Q4 != Principal Engineer. Good for small-to-medium tasks; unreliable on
  large multi-file refactors or subtle architecture work.
- Tool-format reliability across runs varies 87–100% on the 15-prompt probe
  due to model stochasticity (temperature 0.2 doesn't fully eliminate it).
- The model occasionally recites EAS rules from memory instead of calling
  `get_language_rules`. The auto-injected hints in edit/write tool results
  mitigate this but cannot fully eliminate it.
- `/review` is a structured single-pass, not real multi-agent — true 5-Eye
  parallel personas are physically impossible on 7B in one forward pass.
  See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the realist score sheet.

## VRAM management

shadow-qwen needs ~6.5 GB VRAM (Q4_K_M, 32K ctx). Ollama keeps the model
loaded in VRAM for 5 minutes after the last request (default keep-alive).
On 8 GB cards (e.g. GTX 1070) this leaves headroom for one model only.

To unload immediately:

```bash
curl -s http://localhost:11434/api/generate \
  -d '{"model":"shadow-qwen:latest","keep_alive":0}'
```

See [docs/VRAM.md](docs/VRAM.md) for the full guide: status checks,
batch unload, `OLLAMA_KEEP_ALIVE` configuration, ready-to-use shell
aliases (`gpu-status`, `gpu-free`, `gpu-free-all`), and troubleshooting.

## Comparison vs. base shadow-code

| | Base | qwen variant |
|---|---|---|
| Default model | gemma3:27b | shadow-qwen (qwen2.5-coder:7b) |
| Modelfile | gemma stops | ChatML stops + tuned repeat penalty |
| Parser passes | 1 | 4 |
| EAS baseline | none | full 20 zero-trust rules + Iron Laws |
| Rule loader | none | `get_language_rules` + auto-inject |
| Skills | 13 | 20 (planning, review, hardening, distill, api-design, audit, migrate…) |
| Anti-hallucination block | none | yes |
| Bilingual EN/KA preamble | none | auto-prepended to every skill |
