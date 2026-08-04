"""Content-addressed prompt snapshot store (WU-04).

Layout under the store root (default ~/.local/state/shadow-code/prompts):

    <digest>/compiled.txt   -- exact compiled UTF-8 bytes
    <digest>/sources.json   -- provenance + base64 normalized source bytes
    active                  -- file containing the active digest

Snapshots are immutable and content-addressed; the active pointer is
switched atomically (write temp + os.replace) and only after the target
snapshot has been loaded and re-verified from its stored bytes, so any
failure leaves the previous active untouched. A snapshot reproduces its
exact compiled AND normalized source bytes without consulting current
files.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .prompt_compiler import CompiledPrompt, PromptSource


class PromptStoreError(Exception):
    """Typed, visible failure in the snapshot store."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def default_store_dir() -> Path:
    """Store root: $XDG_STATE_HOME/shadow-code/prompts (or ~/.local/state)."""
    state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(state_home) if state_home else Path.home() / ".local" / "state"
    return base / "shadow-code" / "prompts"


def _is_digest(name: str) -> bool:
    return len(name) == 64 and all(char in "0123456789abcdef" for char in name)


@dataclass(frozen=True, slots=True)
class PromptStore:
    root: Path

    @property
    def _active_file(self) -> Path:
        return self.root / "active"

    def save(self, compiled: CompiledPrompt) -> str:
        """Persist a snapshot content-addressed; re-saving is a no-op."""
        digest = compiled.digest
        target = self.root / digest
        if (target / "compiled.txt").is_file() and (target / "sources.json").is_file():
            return digest

        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "digest": digest,
            "created_utc": compiled.created_utc,
            "registry_digest": compiled.registry_digest,
            "sources": [
                {
                    "layer": source.layer,
                    "origin": source.origin,
                    "sha256": source.sha256,
                    "size": source.size,
                    "data_b64": base64.b64encode(
                        compiled.normalized_sources()[source.layer]
                    ).decode("ascii"),
                }
                for source in compiled.sources
            ],
        }
        staging = self.root / f".staging-{digest}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        try:
            (staging / "compiled.txt").write_bytes(compiled.compiled_text.encode("utf-8"))
            (staging / "sources.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            os.replace(staging, target)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return digest

    def _resolve(self, digest_prefix: str) -> str:
        """Resolve a unique digest prefix to a full digest."""
        if not self.root.is_dir():
            raise PromptStoreError("snapshot_not_found", "prompt snapshot store is empty")
        if _is_digest(digest_prefix) and (self.root / digest_prefix).is_dir():
            return digest_prefix
        matches = sorted(
            child.name
            for child in self.root.iterdir()
            if child.is_dir() and _is_digest(child.name) and child.name.startswith(digest_prefix)
        )
        if not matches:
            raise PromptStoreError(
                "snapshot_not_found", f"no prompt snapshot matches '{digest_prefix}'"
            )
        if len(matches) > 1:
            raise PromptStoreError(
                "ambiguous_digest",
                f"digest prefix '{digest_prefix}' matches {len(matches)} snapshots",
            )
        return matches[0]

    def load(self, digest_prefix: str) -> CompiledPrompt:
        """Load and verify a snapshot without consulting current files.

        Stored bytes must re-hash to the snapshot digest and every source
        blob to its recorded sha256; any mismatch fails as corrupt.
        """
        digest = self._resolve(digest_prefix)
        directory = self.root / digest
        try:
            compiled_bytes = (directory / "compiled.txt").read_bytes()
            raw_payload = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PromptStoreError(
                "corrupt_snapshot", f"snapshot {digest[:12]} is unreadable: {exc}"
            ) from exc
        if hashlib.sha256(compiled_bytes).hexdigest() != digest:
            raise PromptStoreError(
                "corrupt_snapshot",
                f"snapshot {digest[:12]} compiled bytes do not re-hash to its digest",
            )

        try:
            sources = tuple(
                PromptSource(
                    layer=str(entry["layer"]),
                    origin=str(entry["origin"]),
                    sha256=str(entry["sha256"]),
                    size=int(entry["size"]),
                )
                for entry in raw_payload["sources"]
            )
            normalized = tuple(
                (source.layer, base64.b64decode(entry["data_b64"], validate=True))
                for source, entry in zip(sources, raw_payload["sources"], strict=True)
            )
            registry_digest = str(raw_payload["registry_digest"])
            created_utc = str(raw_payload["created_utc"])
        except (KeyError, TypeError, ValueError, binascii.Error) as exc:
            raise PromptStoreError(
                "corrupt_snapshot", f"snapshot {digest[:12]} metadata is malformed: {exc}"
            ) from exc

        for source, (layer, data) in zip(sources, normalized, strict=True):
            if len(data) != source.size or hashlib.sha256(data).hexdigest() != source.sha256:
                raise PromptStoreError(
                    "corrupt_snapshot",
                    f"snapshot {digest[:12]} layer {layer} bytes do not match provenance",
                )
        try:
            compiled_text = compiled_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PromptStoreError(
                "corrupt_snapshot", f"snapshot {digest[:12]} compiled bytes are not UTF-8"
            ) from exc
        return CompiledPrompt(
            compiled_text=compiled_text,
            digest=digest,
            sources=sources,
            registry_digest=registry_digest,
            created_utc=created_utc,
            _normalized=normalized,
        )

    def history(self) -> list[CompiledPrompt]:
        """All loadable snapshots, newest first by creation timestamp."""
        if not self.root.is_dir():
            return []
        snapshots = []
        for child in self.root.iterdir():
            if child.is_dir() and _is_digest(child.name):
                try:
                    snapshots.append(self.load(child.name))
                except PromptStoreError:
                    continue  # corrupt snapshots still fail loudly in load()
        snapshots.sort(key=lambda snapshot: snapshot.created_utc, reverse=True)
        return snapshots

    def get_active(self) -> str | None:
        """The active digest, or None when no pointer exists yet."""
        try:
            content = self._active_file.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise PromptStoreError(
                "active_pointer_unreadable", f"cannot read active prompt pointer: {exc}"
            ) from exc
        if not _is_digest(content):
            raise PromptStoreError(
                "invalid_active_pointer",
                f"active prompt pointer does not contain a digest: {content!r}",
            )
        return content

    def set_active(self, digest_prefix: str) -> str | None:
        """Atomically switch the active pointer; returns the previous digest.

        The target snapshot is loaded and re-verified BEFORE the pointer is
        touched, so any failure leaves the previous active unchanged.
        """
        target = self.load(digest_prefix)
        previous = self.get_active()
        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / ".active-staging"
        staging.write_text(target.digest + "\n", encoding="utf-8")
        os.replace(staging, self._active_file)
        return previous
