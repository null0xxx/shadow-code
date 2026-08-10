# Next Session Prompt

ჩასვი ეს prompt შემდეგ სესიაში როგორც არის:

---

გავაგრძელოთ Shadow Code პროექტი (shadow-code-main). შენი მეხსიერებიდან
(engram, project: shadow-code) აღადგინე კონტექსტი — იქ ყველაფერია.

მდგომარეობა: roadmap-ის 14 unit-დან 12 მერჯულია main-ში (290bf8a, 872 ტესტი).
WU-12 (ops polish) დაწყებულია და WIP commit-ად არის დაცული branch
feat/wu-12-ops-polish-ზე (893a87d): shadow_code/ops.py (775 ხაზი) + config/main
wiring — გადაუმოწმებელი, ტესტების გარეშე.

ამოცანა:
1. git checkout feat/wu-12-ops-polish და გადახედე WIP diff-ს.
2. დაასრულე WU-12: გეიტები (pytest, ruff, mypy, bandit), ტესტები
   (tests/unit/test_ops.py + integration), /doctor /backup /restore smoke,
   README. სრული სქოუპი მეხსიერების ჩანაწერშია (topic: architecture/wu-12-ops-polish).
   WIP commit საბოლოოდ ჩაანაცვლე სწორი conventional commit-ით.
3. დაამტკიცე ჩემთან და მიაწოდე ჩვეული flow-ით: issue → PR (type:feature) → CI → merge.
4. შემდეგ ბოლო unit: WU-13 (MCP adapter) — roadmap: docs/architecture-roadmap.md.

Gentle AI flow: coder subagent-ები, თითო ნაბიჯზე ვერიფიკაცია, live smoke
(gemma4-cline:32k), conventional commits, AI attribution-ის გარეშე.

---

## სწრაფი მიმოხილვა (2026-08-10)

- **main**: `290bf8a` = origin/main — 12 PR მერჯული, 872 ტესტი (+5 subtests) მწვანე
- **დასრულებული**: WU-00 … WU-11 (Phase 0–4 სრულად + WU-11 eval matrix)
- **დარჩენილი**: WU-12 (ops polish — WIP), WU-13 (MCP adapter — ბოლო)
- **gates**: `pytest tests -q`, `ruff check/format shadow_code tests`, `mypy shadow_code`, `bandit -c pyproject.toml -r shadow_code/ -q`
- **live smoke**: `SHADOW_MODEL=gemma4-cline:32k .venv/bin/shadow-code` (TUI: `SHADOW_TUI=1`, pty: `script -qec`)
- **eval**: `python -m shadow_code.eval --model gemma4-cline:32k`
