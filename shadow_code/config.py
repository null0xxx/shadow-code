import os


def _env_flag(name: str) -> bool:
    """Return whether an opt-in environment flag is explicitly enabled."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL_NAME = os.environ.get("SHADOW_MODEL", "shadow-gemma:latest")
CONTEXT_WINDOW = int(os.environ.get("SHADOW_CTX", "131072"))  # 128K with FlashAttention+q8
MAX_TOOL_TURNS = 20
MAX_NATIVE_TOOL_TURNS = 4  # read-only admission rounds per user message
MAX_CONSECUTIVE_ERRORS = 5
LEGACY_MARKDOWN_TOOLS = _env_flag("SHADOW_LEGACY_MARKDOWN_TOOLS")
TOOL_OUTPUT_MAX_CHARS = 30_000
BASH_DEFAULT_TIMEOUT = 120
BASH_MAX_TIMEOUT = 600
# Strict mode: deny shell execution entirely when no kernel sandbox
# (bwrap/firejail) is available, instead of running it unconfined.
BASH_STRICT = _env_flag("SHADOW_BASH_STRICT")
# Strict mode: withhold the filesystem-write capability entirely, so the
# policy engine denies write_file/edit_file with CAPABILITY_NOT_GRANTED.
MUTATION_STRICT = _env_flag("SHADOW_MUTATION_STRICT")
MAX_LINES_TO_READ = 2000
INTERACTIVE_CMDS = {"vim", "vi", "nano", "less", "more", "top", "htop", "man"}
BLOCKED_PATHS = {
    "/dev/zero",
    "/dev/urandom",
    "/dev/random",
    "/dev/stdin",
    "/dev/stdout",
    "/dev/stderr",
}
# num_predict: max output tokens per response. Default ~2048 is too low for code generation.
# Set high so the model can write complete files without truncation.
MAX_OUTPUT_TOKENS = int(os.environ.get("SHADOW_MAX_TOKENS", "8192"))

MODEL_OPTIONS = {
    "temperature": 0.3,
    "num_ctx": CONTEXT_WINDOW,
    "num_predict": MAX_OUTPUT_TOKENS,
    "top_k": 40,
    "top_p": 0.9,
    "min_p": 0.05,  # Cut tokens with <5% probability -- improves code quality
    "repeat_penalty": 1.05,
}
# Note: penalize_newline removed -- deprecated in Ollama 0.20+, default is already False

# --- Model Routing ---
# Override via SHADOW_MODEL env var or use these as guidance for model selection.
# The compaction model can be smaller/faster since it only summarizes conversation.
COMPACTION_MODEL = os.environ.get("SHADOW_COMPACTION_MODEL", MODEL_NAME)

# Recommended Ollama models by task complexity:
#
#   Task Type              Recommended Model         Why
#   --------------------   -----------------------   ----------------------------
#   Simple file ops        gemma4:2b, qwen3:4b       Fast, low memory, sufficient
#     (read, ls, grep)                                for tool dispatch
#
#   Standard coding        gemma4:31b, qwen3:14b     Good balance of speed and
#     (edit, debug, test)                             reasoning for most tasks
#
#   Complex reasoning      gemma4:31b, qwen3:32b     Multi-file refactoring,
#     (architecture,       deepseek-r1:32b            architectural decisions,
#      multi-step plans)                              long context synthesis
#
#   Compaction/summary     gemma4:2b, qwen3:4b       Only needs to summarize,
#                                                     not reason deeply
#
# Usage:
#   SHADOW_MODEL=gemma4:2b shadow-code           # lightweight mode
#   SHADOW_MODEL=deepseek-r1:32b shadow-code     # deep reasoning mode
#   SHADOW_COMPACTION_MODEL=gemma4:2b shadow-code # fast compaction
