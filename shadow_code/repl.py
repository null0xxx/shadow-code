"""prompt_toolkit REPL for shadow-code.

Claude Code style input with framed text area:
  - Input box with ╭╮╰╯ rounded border
  - Model name in top border
  - Bottom toolbar with status
  - Tab-completion, history, multiline (Alt+Enter)

Falls back to built-in input() if prompt_toolkit is not installed.
"""

from pathlib import Path

try:
    from prompt_toolkit import Application
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style as PTStyle
    from prompt_toolkit.widgets import Frame, TextArea

    _HAS_PROMPT_TOOLKIT = True
except ImportError:
    _HAS_PROMPT_TOOLKIT = False


# Slash commands for tab completion
_SLASH_COMMANDS = [
    "/help",
    "/clear",
    "/exit",
    "/tokens",
    "/save",
    "/load",
    "/list",
    "/info",
    "/skills",
    "/compact",
    "/version",
    "/history",
    "/cd",
    "/prompt",
]

# prompt_toolkit style
_PT_STYLE = None
if _HAS_PROMPT_TOOLKIT:
    _PT_STYLE = PTStyle.from_dict(
        {
            "frame.border": "#888888",
            "frame.label": "bold #d77757",
            "text-area": "#e0e0e0",
            "toolbar": "bg:#1a1a1a #888888",
            "toolbar.key": "#d77757",
            "completion-menu.completion": "bg:#2d2d2d #e0e0e0",
            "completion-menu.completion.current": "bg:#d77757 #ffffff bold",
        }
    )


def create_prompt_session(state=None):
    """Create input components. Returns a dict with app-creation callable.

    Args:
        state: Optional SessionState for bottom toolbar.

    Returns:
        A dict with 'get_input' callable, or None if prompt_toolkit unavailable.
    """
    if not _HAS_PROMPT_TOOLKIT:
        return None

    history_path = Path("~/.shadow-code/prompt_history").expanduser()
    history_path.parent.mkdir(parents=True, exist_ok=True)

    return {
        "history_path": str(history_path),
        "state": state,
        "history_entries": _load_history(str(history_path)),
    }


def _load_history(path: str) -> list[str]:
    """Load history entries from file."""
    try:
        entries = []
        p = Path(path)
        if p.exists():
            for line in p.read_text().splitlines():
                if line.startswith("+"):
                    entries.append(line[1:])
        return entries[-100:]  # keep last 100
    except OSError:
        return []


def _save_to_history(path: str, text: str):
    """Append one entry to history file."""
    try:
        with open(path, "a") as f:
            f.write(f"\n+{text}\n")
    except OSError:
        pass


def get_input(session, model_name: str = "") -> str | None:
    """Get input. Returns stripped string, None on EOF, empty on interrupt."""
    if session is not None and _HAS_PROMPT_TOOLKIT:
        return _get_input_app(session, model_name)
    return _get_input_fallback(model_name)


def _get_input_app(session_data: dict, model_name: str = "") -> str | None:
    """Claude Code style: framed input box with Application layout."""
    state = session_data.get("state")
    history_path = session_data.get("history_path", "")

    # Build toolbar text
    def get_toolbar():
        parts = []
        if state:
            parts.append(("class:toolbar.key", f" {state.model_name} "))
            parts.append(("class:toolbar", " │ "))
            parts.append(("class:toolbar", state.format_tokens()))
            parts.append(("class:toolbar", " │ "))
        parts.append(("class:toolbar", "Enter: send │ Alt+Enter: newline │ Ctrl+D: exit │ /help "))
        return parts

    # Text area for input
    completer = WordCompleter(_SLASH_COMMANDS, sentence=True)
    text_area = TextArea(
        multiline=True,
        completer=completer,
        style="class:text-area",
        prompt="",
        focus_on_click=True,
        wrap_lines=True,
    )

    # Key bindings
    bindings = KeyBindings()

    @bindings.add("enter")
    def _submit(event):
        """Enter sends the message."""
        text = text_area.text.strip()
        if text:
            event.app.exit(result=text)
        # Empty enter does nothing

    @bindings.add("escape", "enter")
    def _newline(event):
        """Alt+Enter inserts a newline."""
        text_area.buffer.insert_text("\n")

    @bindings.add("c-d")
    def _exit(event):
        event.app.exit(result=None)

    @bindings.add("c-x")
    def _exit2(event):
        event.app.exit(result=None)

    @bindings.add("c-u")
    def _clear(event):
        text_area.text = ""

    @bindings.add("c-l")
    def _clear_screen(event):
        event.app.renderer.clear()

    # Frame with model name in title
    title = model_name or "shadow"
    framed = Frame(
        body=text_area,
        title=title,
        style="class:frame",
    )

    # Layout: frame + toolbar
    toolbar_window = Window(
        content=FormattedTextControl(get_toolbar),
        height=1,
        style="class:toolbar",
    )

    layout = Layout(
        HSplit(
            [
                framed,
                toolbar_window,
            ]
        ),
        focused_element=text_area,
    )

    app: Application[str | None] = Application(
        layout=layout,
        key_bindings=bindings,
        style=_PT_STYLE,
        full_screen=False,
        mouse_support=False,
    )

    try:
        result = app.run()
        if result is not None:
            _save_to_history(history_path, result)
        return str(result) if result is not None else None
    except (EOFError, KeyboardInterrupt):
        return None


def _get_input_fallback(model_name: str = "") -> str | None:
    """Fallback with simple ANSI bordered input."""
    import shutil

    _B = "\033[38;5;240m"  # border color
    _O = "\033[38;5;173m"  # orange
    _R = "\033[0m"
    _TL, _TR, _BL, _BR = "\u256d", "\u256e", "\u2570", "\u256f"
    _H, _V = "\u2500", "\u2502"

    cols = shutil.get_terminal_size().columns
    inner = cols - 2
    label = f" {model_name or 'shadow'} "
    bar = _H * max(0, inner - len(label))

    # Top border
    print(f"{_B}{_TL}{_O}{label}{_B}{bar}{_TR}{_R}")

    try:
        text = input(f"{_B}{_V}{_R} ")
    except EOFError:
        print(f"{_B}{_BL}{_H * inner}{_BR}{_R}")
        return None
    except KeyboardInterrupt:
        print(f"\n{_B}{_BL}{_H * inner}{_BR}{_R}")
        return ""

    # Bottom border
    print(f"{_B}{_BL}{_H * inner}{_BR}{_R}")
    return text.strip()
