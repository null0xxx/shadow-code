"""Layered system-prompt compilation with deterministic digests (WU-04).

Layer order is fixed and deterministic:

1. builtin base      -- code-owned behavioral instructions (render_system_prompt)
2. user overlay      -- ~/.config/shadow-code/prompt.md (optional)
3. workspace overlay -- <workspace>/.shadow-code/prompt.md (optional)
4. tool docs         -- always rendered last from the live registry

Layer 4 is never editable: the registry remains the single source of tool
truth, so prompt text can describe tools but cannot redefine them -- and it
can never grant a capability (policy decisions never read prompt contents).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .prompt import render_system_prompt, render_tool_documentation
from .tools.registry import ToolRegistry

if TYPE_CHECKING:  # prompt_store imports CompiledPrompt from here
    from .prompt_store import PromptStore

MAX_SOURCE_BYTES = 64 * 1024  # per overlay source
MAX_COMPILED_BYTES = 256 * 1024  # whole compiled prompt

LAYER_BUILTIN = "builtin"
LAYER_USER = "user"
LAYER_WORKSPACE = "workspace"
LAYER_TOOLS = "tools"
_LAYER_ORDER = (LAYER_BUILTIN, LAYER_USER, LAYER_WORKSPACE, LAYER_TOOLS)

_TOOLS_SECTION_MARKER = "# Available Tools"


class PromptCompileError(Exception):
    """Typed, visible failure while reading or assembling prompt sources."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def default_user_overlay_path() -> Path:
    """User overlay: $XDG_CONFIG_HOME/shadow-code/prompt.md (or ~/.config)."""
    config_home = os.environ.get("XDG_CONFIG_HOME", "").strip()
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "shadow-code" / "prompt.md"


def default_workspace_overlay_path(workspace_root: str | Path) -> Path:
    """Workspace overlay: <workspace>/.shadow-code/prompt.md."""
    return Path(workspace_root) / ".shadow-code" / "prompt.md"


@dataclass(frozen=True, slots=True)
class PromptSource:
    """Provenance for one compiled layer."""

    layer: str  # builtin | user | workspace | tools
    origin: str  # source path, "builtin", or "registry"
    sha256: str  # sha256 of the normalized source bytes
    size: int  # byte length of the normalized source bytes


@dataclass(frozen=True, slots=True)
class CompiledPrompt:
    """A compiled prompt plus everything needed to reproduce it offline."""

    compiled_text: str
    digest: str  # sha256 of the canonical compiled UTF-8 bytes
    sources: tuple[PromptSource, ...]
    registry_digest: str
    created_utc: str  # ISO timestamp; metadata only, never in compiled bytes
    _normalized: tuple[tuple[str, bytes], ...] = field(repr=False, compare=False)

    def normalized_sources(self) -> dict[str, bytes]:
        """Map layer -> normalized source bytes (CRLF folded to LF)."""
        return dict(self._normalized)


def _read_overlay(path: Path, *, layer: str, max_source_bytes: int) -> bytes | None:
    """Read and normalize one overlay; None when the file does not exist."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PromptCompileError(
            "source_unreadable", f"cannot read {layer} prompt overlay {path}: {exc}"
        ) from exc
    data = raw.replace(b"\r\n", b"\n")
    if len(data) > max_source_bytes:
        raise PromptCompileError(
            "source_too_large",
            f"{layer} prompt overlay {path} is {len(data)} bytes (limit {max_source_bytes})",
        )
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PromptCompileError(
            "source_invalid_utf8", f"{layer} prompt overlay {path} is not valid UTF-8"
        ) from exc
    return data


def _header(layer: str, origin: str, source_sha256: str) -> str:
    return f"\n\n<!-- layer: {layer} source:{origin} sha256:{source_sha256[:12]} -->\n"


def compile_prompt(
    *,
    user_path: Path | None,
    workspace_path: Path | None,
    registry: ToolRegistry,
    legacy_markdown_tools: bool = False,
    now: datetime | None = None,
    max_source_bytes: int = MAX_SOURCE_BYTES,
    max_compiled_bytes: int = MAX_COMPILED_BYTES,
) -> CompiledPrompt:
    """Compile the layered system prompt deterministically.

    Identical inputs produce identical compiled bytes and digest; the
    timestamp lives only in metadata. Missing overlays skip their layer;
    unreadable, non-UTF-8, or oversize overlays fail with PromptCompileError.
    """
    base_text = render_system_prompt(native_tools=True, legacy_markdown_tools=legacy_markdown_tools)
    base_bytes = base_text.encode("utf-8")

    parts = [base_text]
    sources = [PromptSource(LAYER_BUILTIN, "builtin", _sha256(base_bytes), len(base_bytes))]
    normalized: list[tuple[str, bytes]] = [(LAYER_BUILTIN, base_bytes)]

    for layer, path in ((LAYER_USER, user_path), (LAYER_WORKSPACE, workspace_path)):
        if path is None:
            continue
        data = _read_overlay(path, layer=layer, max_source_bytes=max_source_bytes)
        if data is None:
            continue
        origin = str(path)
        sources.append(PromptSource(layer, origin, _sha256(data), len(data)))
        normalized.append((layer, data))
        parts.append(_header(layer, origin, sources[-1].sha256) + data.decode("utf-8"))

    docs_text = render_tool_documentation(registry)
    docs_bytes = docs_text.encode("utf-8")
    docs_sha = _sha256(docs_bytes)
    sources.append(PromptSource(LAYER_TOOLS, "registry", docs_sha, len(docs_bytes)))
    normalized.append((LAYER_TOOLS, docs_bytes))
    parts.append(_header(LAYER_TOOLS, "registry", docs_sha) + docs_text)

    compiled_text = "".join(parts)
    compiled_bytes = compiled_text.encode("utf-8")
    if len(compiled_bytes) > max_compiled_bytes:
        raise PromptCompileError(
            "compiled_too_large",
            f"compiled prompt is {len(compiled_bytes)} bytes (limit {max_compiled_bytes})",
        )

    created = (now or datetime.now(timezone.utc)).isoformat()
    return CompiledPrompt(
        compiled_text=compiled_text,
        digest=_sha256(compiled_bytes),
        sources=tuple(sources),
        registry_digest=registry.digest,
        created_utc=created,
        _normalized=tuple(normalized),
    )


def validate_prompt(
    compiled: CompiledPrompt,
    registry: ToolRegistry | None = None,
    *,
    max_compiled_bytes: int = MAX_COMPILED_BYTES,
) -> list[str]:
    """Structural checks on a compiled prompt; empty list means valid.

    With a live registry, also verifies the embedded tool documentation is
    byte-identical to what the registry renders now (no tool-doc drift).
    """
    issues: list[str] = []
    compiled_bytes = compiled.compiled_text.encode("utf-8")

    if not compiled.compiled_text.strip():
        issues.append("compiled prompt is empty")
    if len(compiled_bytes) > max_compiled_bytes:
        issues.append(
            f"compiled prompt is {len(compiled_bytes)} bytes (limit {max_compiled_bytes})"
        )
    if _sha256(compiled_bytes) != compiled.digest:
        issues.append("compiled bytes do not re-hash to the stored digest")

    layers = [source.layer for source in compiled.sources]
    if layers != [layer for layer in _LAYER_ORDER if layer in layers]:
        issues.append(f"layer order is not deterministic: {layers}")
    if LAYER_BUILTIN not in layers:
        issues.append("builtin base layer is missing")
    if LAYER_TOOLS not in layers:
        issues.append("generated tool documentation layer is missing")
    if _TOOLS_SECTION_MARKER not in compiled.compiled_text:
        issues.append("tool documentation section is missing from compiled text")

    normalized = compiled.normalized_sources()
    for source in compiled.sources:
        data = normalized.get(source.layer)
        if data is None:
            issues.append(f"normalized source bytes missing for layer {source.layer}")
            continue
        if len(data) != source.size:
            issues.append(f"source size mismatch for layer {source.layer}")
        if _sha256(data) != source.sha256:
            issues.append(f"source digest mismatch for layer {source.layer}")

    if registry is not None:
        if compiled.registry_digest != registry.digest:
            issues.append("registry digest mismatch: snapshot predates the live registry")
        elif normalized.get(LAYER_TOOLS) != render_tool_documentation(registry).encode("utf-8"):
            issues.append("tool documentation does not match the live registry")
    return issues


def _overlay_mtimes(paths: tuple[Path, ...]) -> tuple[tuple[int, int] | None, ...]:
    """(mtime_ns, size) per overlay path; None when the file is absent."""
    states: list[tuple[int, int] | None] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            states.append(None)
        else:
            states.append((stat.st_mtime_ns, stat.st_size))
    return tuple(states)


@dataclass
class PromptManager:
    """Holds the active compiled prompt and watches overlay sources.

    The audit trail is a printed line plus the snapshot history; the durable
    event store arrives with WU-06.
    """

    registry: ToolRegistry
    store: PromptStore
    user_path: Path
    workspace_path: Path
    legacy_markdown_tools: bool
    active: CompiledPrompt
    _watched: tuple[tuple[int, int] | None, ...] = field(default_factory=tuple)

    @classmethod
    def bootstrap(
        cls,
        *,
        registry: ToolRegistry,
        store: PromptStore,
        user_path: Path,
        workspace_path: Path,
        legacy_markdown_tools: bool = False,
    ) -> tuple[PromptManager, str | None]:
        """Compile from current sources, save, and activate when changed.

        Returns (manager, previous_active_digest); the previous digest is
        None on first run or when the active snapshot was already current,
        so the caller can print the audit line only on a real switch.
        """
        manager = cls(
            registry=registry,
            store=store,
            user_path=user_path,
            workspace_path=workspace_path,
            legacy_markdown_tools=legacy_markdown_tools,
            active=cls._compile(registry, user_path, workspace_path, legacy_markdown_tools),
        )
        manager.store.save(manager.active)
        previous = manager.store.get_active()
        if previous != manager.active.digest:
            manager.store.set_active(manager.active.digest)
        else:
            previous = None
        manager._watched = _overlay_mtimes(manager._overlay_paths())
        return manager, previous

    @staticmethod
    def _compile(
        registry: ToolRegistry,
        user_path: Path,
        workspace_path: Path,
        legacy_markdown_tools: bool,
    ) -> CompiledPrompt:
        return compile_prompt(
            user_path=user_path,
            workspace_path=workspace_path,
            registry=registry,
            legacy_markdown_tools=legacy_markdown_tools,
        )

    def _overlay_paths(self) -> tuple[Path, ...]:
        return (self.user_path, self.workspace_path)

    def _activate(self, compiled: CompiledPrompt) -> str | None:
        """Save + atomically activate; returns the previous active digest."""
        previous = self.active.digest
        self.store.save(compiled)
        self.store.set_active(compiled.digest)
        self.active = compiled
        self._watched = _overlay_mtimes(self._overlay_paths())
        return previous if previous != compiled.digest else None

    def watch(self) -> tuple[CompiledPrompt, str | None] | None:
        """Recompile and activate when an overlay changed; None when not.

        On any compile/store failure this raises and the active snapshot
        stays untouched, so a broken edit never takes effect mid-session.
        """
        current = _overlay_mtimes(self._overlay_paths())
        if current == self._watched:
            return None
        compiled = self._compile(
            self.registry, self.user_path, self.workspace_path, self.legacy_markdown_tools
        )
        previous = self._activate(compiled)
        return compiled, previous

    def reload(self) -> tuple[CompiledPrompt, str | None]:
        """Force a recompile and activation, regardless of mtimes."""
        compiled = self._compile(
            self.registry, self.user_path, self.workspace_path, self.legacy_markdown_tools
        )
        previous = self._activate(compiled)
        return compiled, previous

    def rollback(self, digest_prefix: str) -> tuple[CompiledPrompt, str | None]:
        """Verify + validate a stored snapshot, then switch active atomically.

        Any failure raises a typed error BEFORE the active pointer changes,
        so a failed rollback always leaves the previous active in place.
        """
        from .prompt_store import PromptStoreError

        target = self.store.load(digest_prefix)  # verifies stored bytes/digests
        if target.registry_digest != self.registry.digest:
            raise PromptStoreError(
                "incompatible_snapshot",
                f"snapshot {target.digest[:12]} was compiled against a different "
                "tool registry; refusing to activate",
            )
        issues = validate_prompt(target, self.registry)
        if issues:
            raise PromptStoreError(
                "invalid_snapshot",
                f"snapshot {target.digest[:12]} failed validation: " + "; ".join(issues),
            )
        previous = self.store.set_active(target.digest)
        self.active = target
        return target, previous
