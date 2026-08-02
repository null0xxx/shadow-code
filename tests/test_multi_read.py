"""Read-only multi-file behavior tests confined to temporary workspaces."""

import builtins
from pathlib import Path

import pytest

from shadow_code.tool_context import ToolContext
from shadow_code.tools import multi_read
from shadow_code.tools.multi_read import MultiReadTool


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"paths": []},
        {"paths": "not-a-list"},
        {"paths": ["relative.txt"]},
        {"paths": [f"/tmp/{index}" for index in range(11)]},
    ],
)
def test_multi_read_rejects_malformed_requests(tmp_path: Path, params: dict) -> None:
    error = MultiReadTool(ToolContext(str(tmp_path))).validate(params)
    assert error is not None


def test_multi_read_reports_mixed_workspace_inputs_and_tracks_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = tmp_path / "text.txt"
    text.write_text("one\ntwo\nthree\n", encoding="utf-8")
    latin = tmp_path / "latin.txt"
    latin.write_bytes("olá\n".encode("latin-1"))
    binary = tmp_path / "binary.bin"
    binary.write_bytes(b"prefix\x00suffix")
    directory = tmp_path / "directory"
    directory.mkdir()
    missing = tmp_path / "missing.txt"
    blocked = tmp_path / "blocked.txt"
    blocked.write_text("must not be read", encoding="utf-8")
    monkeypatch.setattr(multi_read, "BLOCKED_PATHS", (str(blocked),))
    ctx = ToolContext(str(tmp_path))
    tool = MultiReadTool(ctx)

    result = tool.execute(
        {
            "paths": [
                str(text),
                str(latin),
                str(binary),
                str(directory),
                str(missing),
                str(blocked),
            ],
            "limit": 2,
        }
    )

    assert result.success is True
    assert "one" in result.output and "two" in result.output
    assert "1 more lines not shown" in result.output
    assert "olá" in result.output
    assert "[BINARY FILE]" in result.output
    assert "[IS A DIRECTORY]" in result.output
    assert "[NOT FOUND]" in result.output
    assert "[BLOCKED PATH]" in result.output
    assert ctx.was_file_read(str(text))
    assert ctx.was_file_read(str(latin))
    assert not ctx.was_file_read(str(blocked))


def test_multi_read_normalizes_binary_probe_errors_inside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    denied = tmp_path / "denied.txt"
    denied.write_text("content", encoding="utf-8")
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if Path(file) == denied and mode == "rb":
            raise PermissionError("denied by test")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)

    result = MultiReadTool(ToolContext(str(tmp_path))).execute({"paths": [str(denied)]})

    assert result.success is True
    assert "[PERMISSION DENIED]" in result.output
