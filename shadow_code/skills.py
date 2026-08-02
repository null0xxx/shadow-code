"""Skill system for shadow-code.

Skills are pre-built prompt templates activated by /command.
Adapted from Claude Code's bundled skills system.

Usage in REPL:
    /commit          -- create a git commit
    /simplify        -- review changed code for quality
    /review          -- review a specific file
    /explain         -- explain code in detail
    /debug           -- debug an error or issue
    /stuck           -- get help when stuck

Each skill injects a detailed prompt that guides the model
through a specific workflow with the available tools.
"""

# Skill registry: name -> (description, prompt_template)
_SKILLS: dict[str, tuple[str, str]] = {}


_BILINGUAL_PREAMBLE = (
    "Respond in user's language (Georgian if user wrote Georgian; "
    "identifiers/paths/code/tool calls always English).\n\n"
)


def register_skill(name: str, description: str, prompt: str):
    """Register a skill with its prompt template.

    Auto-prepends a bilingual response directive unless already present.
    """
    if "Respond in user's language" not in prompt:
        prompt = _BILINGUAL_PREAMBLE + prompt
    _SKILLS[name] = (description, prompt)


def get_skill(name: str) -> tuple[str, str] | None:
    """Get skill (description, prompt) by name, or None if not found."""
    return _SKILLS.get(name)


def list_skills() -> list[tuple[str, str]]:
    """Return list of (name, description) for all registered skills."""
    return [(name, desc) for name, (desc, _) in sorted(_SKILLS.items())]


# ==========================================================================
# Built-in Skills (adapted from Claude Code bundled skills)
# ==========================================================================

register_skill(
    "commit",
    "Create a git commit with staged changes",
    """Create a git commit. Follow these steps:

1. Run git status and git diff to see all changes (staged and unstaged).
2. Run git log --oneline -5 to see recent commit message style.
3. Analyze the changes and draft a concise commit message:
   - Focus on WHY, not WHAT
   - 1-2 sentences maximum
   - Match the repository's existing commit style
4. Do NOT commit files that likely contain secrets (.env, credentials.json)
5. Stage specific files by name (not git add -A or git add .)
6. Create the commit

Git Safety:
- NEVER skip hooks (--no-verify)
- NEVER amend unless explicitly asked
- If pre-commit hook fails, fix the issue and create a NEW commit
- Do NOT push unless explicitly asked""",
)

register_skill(
    "simplify",
    "Review changed code for reuse, quality, and efficiency",
    """Review all changed files for code quality. Follow these steps:

1. Run git diff (or git diff HEAD for staged changes) to see what changed.
2. For each change, check:

Code Reuse:
- Search for existing utilities that could replace new code
- Flag duplicated functionality
- Flag inline logic that could use existing helpers

Code Quality:
- Redundant state or unnecessary caching
- Copy-paste with slight variation (should be unified)
- Leaky abstractions
- Unnecessary comments (WHAT not WHY)

Efficiency:
- Redundant computations or repeated file reads
- N+1 patterns
- Unnecessary string concatenation in loops
- Large objects copied when a reference would work

3. Fix any issues found directly. Don't just report -- fix them.""",
)

register_skill(
    "review",
    "Structured code review (4 sections: ARCHITECT/SECURITY/QA/LEAD + V1-V7)",
    """Respond in user's language (Georgian if user wrote Georgian; identifiers/paths/code always \
English).

Review the specified file or recently changed code. Single-pass structured review.

STEP 1 — Scope:
- Run `git diff --name-only HEAD~1` and `git diff --stat` to identify changes
- Read each changed file completely with read_file
- For language-specific rules call get_language_rules with the file extension
- For security checklist call get_language_rules with name="common-security"

STEP 2 — Output EXACTLY these 4 sections (markdown headers, in order):

## ARCHITECT
- Module boundaries, blast radius, design fit
- Circular deps, layering violations
- Does change fit existing patterns?

## SECURITY
- Input validation (every external input)
- No eval/innerHTML/exec, no secrets in code
- Parameterized queries, path-traversal protection
- AuthN/AuthZ correctness on every endpoint
- Apply common-security rules

## QA
- Edge cases: empty, max, special chars (Georgian!), null/undefined, concurrency
- Regression risk: what existing behavior could break?
- Test coverage gaps for this change

## TECH LEAD VERDICT
- Decision: SHIP / FIX / BLOCK + 1-2 sentence rationale
- V1-V7 status checklist:
  - V1 build/compile: ✅/❌/N/A
  - V2 type safety (no any/ts-ignore): ✅/❌/N/A
  - V3 security (no eval/innerHTML/secrets): ✅/❌/N/A
  - V4 code quality (≤50 lines/fn, ≤300 lines/file, no TODO/FIXME): ✅/❌/N/A
  - V5 spec compliance: ✅/❌/N/A
  - V6 regression safety: ✅/❌/N/A
  - V7 edge cases handled: ✅/❌/N/A

Report individual findings inline: `file:line — issue → suggested fix`.
Critical issues block; minor ones become follow-up notes.""",
)

register_skill(
    "explain",
    "Explain code in detail",
    """Explain the specified code file or function in detail.

1. Read the file with read_file
2. Explain:
   - Purpose and high-level design
   - Key functions and their roles
   - Data flow and control flow
   - Dependencies and interactions with other modules
   - Any non-obvious logic or clever patterns
3. Use clear language. If the user writes in Georgian, explain in Georgian.
   Keep technical terms in English.""",
)

register_skill(
    "debug",
    "Debug an error or issue",
    """Help debug the reported error or issue.

1. Read any error messages or logs the user provided
2. Read the relevant source files with read_file
3. Search for related code with grep if needed
4. Identify the root cause:
   - Check the error location (file:line)
   - Trace the data flow that leads to the error
   - Check edge cases and error handling
5. Fix the issue with edit_file
6. Suggest how to verify the fix (run test, execute script, etc.)""",
)

register_skill(
    "stuck",
    "Get help when stuck on a problem",
    """The user is stuck. Help them get unstuck.

1. Ask clarifying questions if the problem is unclear
2. Read relevant code with read_file
3. Search the codebase with grep/glob for related patterns
4. Consider multiple approaches:
   - What's the simplest solution?
   - What does the existing codebase do in similar cases?
   - Are there any constraints or requirements being missed?
5. Propose a concrete next step with specific code changes""",
)

register_skill(
    "pr",
    "Create a pull request",
    """Create a pull request. Follow these steps:

1. Run these commands to understand the branch:
   - git status (see all changes)
   - git diff main...HEAD (or appropriate base branch)
   - git log main..HEAD --oneline (all commits on this branch)

2. Analyze ALL commits (not just the latest) and draft:
   - PR title: short, under 70 characters
   - PR body: summary (1-3 bullets), test plan (checklist)

3. Push and create PR:
   - Push with -u flag if needed
   - Use gh pr create with the title and body

Important:
- Look at ALL commits, not just the most recent
- Return the PR URL when done""",
)

register_skill(
    "remember",
    "Save important information for future reference",
    """The user wants to save something for future reference.

1. Understand what they want to remember:
   - A decision that was made and why
   - A pattern or convention used in this project
   - A bug fix approach that worked
   - A configuration detail
   - User preferences for how to work

2. Create or update a memory file:
   - Save to ~/.shadow-code/memory/ directory
   - Use descriptive filenames (e.g., "db-migration-pattern.md", "user-preferences.md")
   - Include: WHAT, WHY, and WHEN (date)
   - Keep it concise but complete enough to be useful later

3. If the user says "remember that..." or "don't forget...", extract the key
   information and save it without asking unnecessary questions.

4. If asked "what do you remember about X?", search the memory directory
   with glob and grep for relevant files.""",
)

register_skill(
    "verify",
    "Verify that code changes actually work",
    """Verify that recent code changes work correctly.

1. Identify what was changed:
   - Run git diff to see recent changes
   - Or check the conversation for files that were edited

2. For each change, determine how to verify:
   - If tests exist: run them (e.g., pytest, npm test, go test)
   - If it's a script: execute it and check output
   - If it's a server: start it and check it responds
   - If it's a build: run the build and check for errors

3. Run the verification:
   - Execute the appropriate test/build/run command
   - Capture and analyze the output
   - Report: PASS (with evidence) or FAIL (with error details)

4. If verification fails:
   - Identify the root cause
   - Fix the issue
   - Re-verify

IMPORTANT: Actually run the verification. Do not claim "should work" --
prove it works by executing code and showing output.""",
)

register_skill(
    "init",
    "Initialize shadow-code for a new project",
    """Set up shadow-code for working with a new project directory.

1. Explore the project structure:
   - Run list_dir to see top-level files and directories
   - Look for configuration files (package.json, pyproject.toml, Makefile, etc.)
   - Look for README.md or documentation
   - Check if it's a git repository (git status)

2. Identify the project type:
   - Language (Python, JavaScript/TypeScript, Go, Rust, etc.)
   - Framework (React, Django, FastAPI, etc.)
   - Build system (npm, pip, cargo, make, etc.)
   - Test framework (pytest, jest, go test, etc.)

3. Report findings to the user:
   - Project type and language
   - Key directories and their purpose
   - Available commands (build, test, run)
   - Any issues found (missing dependencies, broken config)

4. Suggest next steps based on what was found.""",
)

register_skill(
    "test",
    "Run tests and analyze results (testing pyramid: unit > integration > e2e)",
    """Run the project's test suite and analyze results following the testing pyramid:
unit (many, fast) > integration (moderate) > e2e (few, slow).

1. Detect the test framework:
   - Python: pytest.ini / pyproject.toml [tool.pytest]
   - JS/TS: jest.config / vitest.config / package.json scripts
   - Go: `go test ./...`
   - Rust: `cargo test`
   - E2E specifically: playwright.config.ts → use /test --e2e flow

2. If user passes `--e2e`: focus on Playwright. Check for page objects in
   tests/e2e/, run `npx playwright test`, capture trace/screenshot artifacts on
   failure, identify flaky tests (look for `.only`, hardcoded waits, retries).

3. Run tests:
   - Use the appropriate command, capture full output
   - For unit: high parallelism. For e2e: serial, with --reporter=list

4. Analyze:
   - Count passed/failed/skipped
   - For each failure: read the test file AND the code under test
   - Distinguish test bugs from production bugs
   - Check for missing edge cases (empty, max, special chars, null, concurrency)

5. Coverage check (if requested):
   - pytest --cov / vitest --coverage / go test -cover
   - Flag any function/branch < 80% coverage in changed files

6. Fix and re-run. Don't claim success without seeing green output.""",
)

register_skill(
    "refactor",
    "Refactor code while preserving behavior",
    """Refactor the specified code (file, function, or module).

1. Read and understand the code completely with read_file
2. Identify what to improve:
   - Long functions that should be split
   - Duplicated code that should be extracted
   - Complex conditionals that should be simplified
   - Poor naming that should be clarified
   - Tight coupling that should be loosened

3. Plan the refactoring:
   - List each change you'll make
   - Ensure behavior is preserved (no functional changes)
   - Consider: will existing tests still pass?

4. Execute changes with edit_file:
   - Make one logical change at a time
   - After each change, verify the code is still valid

5. Verify:
   - If tests exist, run them
   - If no tests, explain what you changed and why it's safe""",
)

register_skill(
    "search",
    "Deep search across the codebase",
    """Perform a thorough search across the codebase.

1. Understand what the user is looking for:
   - A specific function, class, or variable?
   - A pattern or convention?
   - Where something is used?
   - How something is implemented?

2. Use multiple search strategies:
   - grep for exact text matches
   - glob for file name patterns
   - grep with regex for pattern matching
   - Read key files that are likely relevant

3. Report findings:
   - List all matches with file_path:line_number
   - Show relevant code context
   - Explain how the pieces connect

4. If searching for usage: trace the call chain from definition to all callers.""",
)


register_skill(
    "audit",
    "Security audit (STRIDE + automated scans + manual review)",
    """Run a security audit on the codebase or a specific feature.

Phase 1 — Automated scans (run via bash):
- Dependencies: `npm audit --audit-level=moderate` / `pip-audit` / `cargo audit`
- Secret scan: grep -rEn "(password|secret|token|api_key|apikey)\\s*[=:]\\s*['\"]" .
- Private keys: grep -rn "-----BEGIN.*PRIVATE KEY-----" .
- AWS keys: grep -rEn "AKIA[0-9A-Z]{16}" .
- .env in .gitignore? `grep -q '.env' .gitignore`

Phase 2 — STRIDE manual review for each component:
- Spoofing: are identities verified? auth checks on every endpoint?
- Tampering: integrity of inputs, signed tokens, constant-time comparisons?
- Repudiation: audit logging on sensitive ops?
- Information disclosure: error messages leak stack traces / internals?
- Denial of service: rate limiting, input size caps, recursion depth?
- Elevation of privilege: authorization checks AFTER auth, principle of least privilege?

Phase 3 — Code-level checks (call get_language_rules name="common-security" first):
- All inputs validated (whitelist, not blacklist)
- No eval/exec/innerHTML/dangerouslySetInnerHTML
- Parameterized queries (no string concat for SQL)
- Path traversal protection on file ops
- Crypto: no MD5/SHA1, no ECB mode, no hardcoded IVs

Report each finding as: `severity (CRITICAL/HIGH/MEDIUM/LOW) | file:line | issue | fix`.""",
)


register_skill(
    "migrate",
    "Database migration (safe, reversible, zero-downtime)",
    """Plan or implement a database migration safely.

Core principles:
1. Every change is a migration — never alter production manually
2. Migrations are forward-only in production (rollback = new forward migration)
3. Schema and data migrations are SEPARATE files
4. Test against production-sized data, not toy datasets
5. Migrations are immutable once deployed

Pre-flight checklist:
- [ ] UP and DOWN both written (or explicitly marked irreversible)
- [ ] No full table locks on large tables (use CONCURRENT operations)
- [ ] New columns are nullable OR have defaults (never NOT NULL without default)
- [ ] Indexes created CONCURRENTLY (Postgres) — separate from CREATE TABLE
- [ ] Backfill is a separate migration, batched (1k-10k rows per batch)
- [ ] Rollback plan documented in the migration file header

PostgreSQL safe patterns:
- Add nullable column: `ALTER TABLE t ADD COLUMN c TEXT;` — instant
- Add column with default (PG11+): `ALTER TABLE t ADD COLUMN c BOOL NOT NULL \
DEFAULT true;` — instant
- Add NOT NULL to existing column: 1) add CHECK constraint NOT VALID, 2) backfill, \
3) VALIDATE CONSTRAINT, 4) SET NOT NULL
- Index: `CREATE INDEX CONCURRENTLY idx_t_c ON t(c);`

ORM-specific notes (detect from project):
- Prisma: `prisma migrate dev` (dev) / `prisma migrate deploy` (prod)
- Drizzle: `drizzle-kit generate` then commit
- Django: separate `migrations/0001_schema.py` and `migrations/0002_backfill.py`
- golang-migrate: paired `.up.sql` / `.down.sql`

After writing the migration:
1. Run it against a local copy of prod-shaped data
2. Time it; flag any DDL > 1s on a production-sized table
3. Verify the DOWN migration also works""",
)


register_skill(
    "plan",
    "Produce a concrete implementation plan (no edits)",
    """Respond in user's language (Georgian if user wrote Georgian; identifiers/paths/code always \
English).

Produce a concrete implementation plan for the user's request. Plan ONLY — do NOT
execute any edits, do NOT call write_file/edit_file/bash. Stop after the plan.

Output schema (markdown, in order):

## Goal
One sentence: what the user wants accomplished.

## Files affected
- Absolute paths, one per line
- Mark each as NEW / MODIFY / DELETE

## Steps
Numbered, ≤6 steps, each ≤2 sentences. Include the tool you'll use:
1. [tool] Brief action description.
2. ...

## Risks
- Top 2-3 risks with one-line mitigation each.

## Verification (V1-V7)
- V1 build: how will you confirm compile passes?
- V2 types: type safety check
- V3 security: any input validation / secret check needed?
- V6 regression: which existing tests/flows must still pass?
- V7 edges: empty/max/special chars/null cases to test

If the request is ambiguous, ask 1-2 clarifying questions instead of guessing.
Cap plan at 6 steps — if more are needed, the request needs to be split.""",
)


register_skill(
    "cross-validate",
    "Adversarial self-review of a design or solution (2-pass structured)",
    """Apply targeted adversarial validation. Output exactly two sections:

## Pass 1 — Proposal
Present your design/solution. Include:
- The approach in 3-5 bullets
- Key assumptions (numbered)
- Files/modules affected

## Pass 2 — Adversarial Self-Review
Now switch hats. Attack your Pass 1 design:
- List 3-5 concrete failure modes with EXAMPLES (not vague concerns):
  * Edge case that breaks it
  * Race / concurrency issue
  * Security exploit angle
  * Backwards-compat / regression risk
  * Operational hazard (rollback, monitoring, etc.)
- For each: severity (CRITICAL / HIGH / MED / LOW)
- Then revise the proposal: list specific changes to address CRITICAL+HIGH only

Use this skill on:
- Auth, crypto, migrations, deletion logic
- Anything affecting >3 files
- Anything user flagged as risky
Stop after Pass 2. Do NOT execute any edits.""",
)


register_skill(
    "harden",
    "Improve resilience: error handling, edge cases, i18n, overflow",
    """Make the specified code production-resilient. Step through:

1. READ — read the target file(s) completely.

2. AUDIT for these failure classes (be specific to the language; call
   get_language_rules with the file extension if needed):
   - Inputs: empty string, max length, unicode (Georgian!), null/undefined, NaN, negative, zero
   - I/O: file not found, permission denied, disk full, network timeout, partial reads
   - Concurrency: race conditions, TOCTOU, double-execution, idempotency
   - Resources: leak on error path, unbounded memory/CPU, missing cleanup in finally/defer
   - i18n: hardcoded English strings in user-facing paths, locale-dependent date/number formatting
   - Overflow: int/float bounds, string truncation, buffer limits
   - State: partial writes leaving inconsistent state, missing rollback

3. FIX with edit_file. Each fix:
   - Add the smallest change that handles the failure correctly
   - Add a test asserting the new behavior (call /test if no harness exists)
   - Prefer fail-closed over fail-open (especially for auth/access checks)

4. VERIFY — run the test suite (bash). Confirm no regression.

Report a brief summary: what was hardened, what tests were added.""",
)


register_skill(
    "distill",
    "Simplify code: remove unnecessary complexity while preserving behavior",
    """Strip the specified code to its essence. Behavior MUST be preserved exactly.

1. READ the target file(s) completely.

2. IDENTIFY waste:
   - Dead code (unreached branches, unused imports/vars/functions)
   - Speculative abstraction (1-use interface, premature generic, "just in case" config)
   - Duplication that should be a helper (DRY)
   - Over-deep nesting → early returns / guard clauses
   - Comments that restate the code instead of explaining WHY
   - Indirection layers with no value (1-line wrappers around stdlib)
   - Magic numbers → named constants (or remove if obvious)

3. REFACTOR with edit_file. Each change:
   - Single-purpose commit (one type of simplification per edit)
   - Behavior must be preserved — DO NOT change semantics
   - Function ≤ 50 lines, file ≤ 300 lines

4. VERIFY — re-run tests (bash) to confirm zero behavior change.

Report: lines removed, abstractions collapsed, what was preserved.
If a simplification would change behavior, STOP and ask the user.""",
)


register_skill(
    "api-design",
    "Design or review a REST/HTTP API for production",
    """Design or audit the specified API. Apply REST production patterns.

If designing a new endpoint, output:

## Resource & Method
- Path: `/api/v1/<resource>` — plural noun, kebab-case
- Method: GET (read) / POST (create) / PUT (full replace) / PATCH (partial) / DELETE
- Idempotency: which methods are idempotent? PUT/DELETE/GET yes; POST may need Idempotency-Key

## Request
- Headers: Authorization, Content-Type, Idempotency-Key (POST), If-Match (optimistic locking)
- Body schema (typed, with required vs optional)
- Validation rules at the boundary

## Response
- Success status: 200 (read) / 201 (created, with Location header) / 204 (no content)
- Body schema (consistent envelope across endpoints)
- Pagination: cursor-based preferred (offset only for small datasets); return next/prev cursors

## Errors
- 400 validation, 401 auth missing, 403 forbidden, 404 not found, 409 conflict, 410 gone, \
422 unprocessable, 429 rate limit, 5xx server
- Error envelope: `{ "error": { "code": "INVALID_INPUT", "message": "...", "details": [...] } }`
- Never leak internals (stack traces, DB errors)

## Cross-cutting
- Versioning: URL `/v1/` or `Accept-Version` header
- Rate limiting: per-token, document limit + Retry-After
- CORS: explicit origins, never wildcard with credentials
- Audit logging: who did what, when

If reviewing existing API, check each section above and flag deviations.""",
)
