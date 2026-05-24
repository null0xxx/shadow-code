#!/usr/bin/env python3
"""Phase 0 — Tool-format validation for qwen2.5-coder.

Sends N fixed prompts directly to Ollama API using shadow-code's system prompt,
parses each response with shadow-code's actual parser, measures success rate.

Decision gate: >=80% pass = stay with markdown ```tool_call``` format.
Else = switch to native Ollama tools API.
"""
import json
import sys
import time
from pathlib import Path

# Add shadow-code to path
ROOT = Path("/home/null0xxx/Documents/projects/shadow-code")
sys.path.insert(0, str(ROOT))

import requests

from shadow_code.prompt import SYSTEM_PROMPT
from shadow_code.parser import parse_tool_calls

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "shadow-qwen:latest"

# 10 prompts that REQUIRE a tool call (positive cases)
# + 5 prompts that should NOT use tools (negative cases)
TESTS = [
    # POSITIVE: must produce a valid tool_call
    {"prompt": "List the files in /tmp", "expect": "tool", "tool": "bash"},
    {"prompt": "Read /etc/hostname for me", "expect": "tool", "tool": "read_file"},
    {"prompt": "Find all Python files in /home/null0xxx/Documents/projects/shadow-code", "expect": "tool", "tool": "glob"},
    {"prompt": "Search for the string 'def main' in *.py files in /home/null0xxx/Documents/projects/shadow-code", "expect": "tool", "tool": "grep"},
    {"prompt": "Show me what's in /home/null0xxx/Documents/projects/shadow-code", "expect": "tool", "tool": "list_dir"},
    {"prompt": "Run the command 'date' and tell me the output", "expect": "tool", "tool": "bash"},
    {"prompt": "Read /home/null0xxx/Documents/projects/shadow-code/README.md", "expect": "tool", "tool": "read_file"},
    {"prompt": "Find all *.toml files in the current project at /home/null0xxx/Documents/projects/shadow-code", "expect": "tool", "tool": "glob"},
    {"prompt": "List files in /home/null0xxx/Documents/projects/shadow-code/shadow_code", "expect": "tool", "tool": "list_dir"},
    {"prompt": "Use bash to print 'hello world'", "expect": "tool", "tool": "bash"},
    # NEGATIVE: must NOT produce a tool_call (just answer)
    {"prompt": "What is 2+2?", "expect": "no_tool"},
    {"prompt": "Explain what a Python decorator is in 2 sentences.", "expect": "no_tool"},
    {"prompt": "Is JavaScript single-threaded?", "expect": "no_tool"},
    {"prompt": "What does 'idempotent' mean?", "expect": "no_tool"},
    {"prompt": "Hello, how are you?", "expect": "no_tool"},
]


def run_one(prompt: str) -> tuple[str, float]:
    """Send prompt, return (response_text, latency_seconds)."""
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_ctx": 32768,
        },
    }
    t0 = time.time()
    r = requests.post(OLLAMA_URL, json=body, timeout=300)
    r.raise_for_status()
    elapsed = time.time() - t0
    return r.json()["message"]["content"], elapsed


def evaluate(case: dict, response: str) -> tuple[bool, str]:
    """Check whether response matches expected outcome."""
    clean, calls = parse_tool_calls(response)
    valid_calls = [c for c in calls if c.tool != "__invalid__"]
    invalid_calls = [c for c in calls if c.tool == "__invalid__"]

    if case["expect"] == "tool":
        if not calls:
            return False, "no tool call produced"
        if invalid_calls:
            return False, f"invalid JSON in tool_call: {invalid_calls[0].params.get('error', '')[:80]}"
        if not valid_calls:
            return False, "no valid tool calls"
        # Soft check: any tool acceptable as long as it's valid format
        actual_tools = [c.tool for c in valid_calls]
        return True, f"OK tools={actual_tools} (expected~{case.get('tool')})"
    else:  # no_tool
        if calls:
            return False, f"unexpected tool call: {[c.tool for c in calls]}"
        return True, "OK no tools"


def main():
    print(f"=== Phase 0 Tool-Format Validation ===")
    print(f"Model:  {MODEL}")
    print(f"System prompt: {len(SYSTEM_PROMPT)} chars")
    print(f"Test cases: {len(TESTS)}\n")

    results = []
    for i, case in enumerate(TESTS, 1):
        print(f"[{i:2}/{len(TESTS)}] {case['expect']:7} | {case['prompt'][:60]:<60}", end=" ", flush=True)
        try:
            resp, lat = run_one(case["prompt"])
            ok, detail = evaluate(case, resp)
            mark = "✅" if ok else "❌"
            print(f"{mark} {lat:5.1f}s | {detail}")
            results.append({"idx": i, "case": case, "ok": ok, "detail": detail, "lat": lat, "resp": resp})
        except Exception as e:
            print(f"💥 ERROR: {e}")
            results.append({"idx": i, "case": case, "ok": False, "detail": f"exception: {e}", "lat": 0, "resp": ""})

    passed = sum(1 for r in results if r["ok"])
    total = len(results)
    rate = passed / total * 100
    avg_lat = sum(r["lat"] for r in results if r["lat"]) / max(1, sum(1 for r in results if r["lat"]))

    print(f"\n=== RESULTS ===")
    print(f"Passed: {passed}/{total} ({rate:.0f}%)")
    print(f"Avg latency: {avg_lat:.1f}s")
    print(f"\nDecision gate: {'PASS — stay with markdown' if rate >= 80 else 'FAIL — switch to native tools API'}")

    # Dump failures
    fails = [r for r in results if not r["ok"]]
    if fails:
        print(f"\n--- {len(fails)} FAILURES ---")
        for r in fails:
            print(f"\n[{r['idx']}] {r['case']['prompt']}")
            print(f"  reason: {r['detail']}")
            print(f"  resp: {r['resp'][:400]}")

    # Save full report
    out = ROOT / "phase0_report.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull report: {out}")
    return 0 if rate >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
