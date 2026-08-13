#!/usr/bin/env bash
# scripts/install.sh -- single local install/update/uninstall command (WU-12).
#
# Personal-CLI scale: one owner, one Linux machine, no sudo. Install creates
# (or updates) a dedicated venv under the XDG data dir, pip-installs this
# checkout into it, and drops a `shadow-code` launcher on ~/.local/bin.
# Uninstall removes the venv and the launcher but PRESERVES user data
# (sessions, events, prompt snapshots, backups, prompt overlays); only
# `uninstall --purge` deletes that data, behind an interactive confirmation.
#
# Usage:
#   scripts/install.sh [install|update]   install or update (default)
#   scripts/install.sh uninstall          remove executable + venv, keep data
#   scripts/install.sh uninstall --purge  also delete user data (asks first)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/shadow-code"
VENV_DIR="$INSTALL_DIR/venv"
BIN_DIR="$HOME/.local/bin"
LAUNCHER="$BIN_DIR/shadow-code"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/shadow-code"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/shadow-code"

usage() {
    sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

die() {
    echo "error: $*" >&2
    exit 1
}

install() {
    local python
    python="$(command -v python3)" || die "python3 not found on PATH"
    "$python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
        || die "python 3.10+ is required ($("$python" --version 2>&1) found)"

    if [[ ! -x "$VENV_DIR/bin/pip" ]]; then
        echo "creating venv: $VENV_DIR"
        "$python" -m venv "$VENV_DIR"
    fi
    echo "installing shadow-code from $REPO_ROOT"
    "$VENV_DIR/bin/pip" install --quiet --upgrade "${REPO_ROOT}[full]"

    mkdir -p "$BIN_DIR"
    {
        printf '#!/usr/bin/env bash\n'
        printf 'exec "%s/bin/shadow-code" "$@"\n' "$VENV_DIR"
    } > "$LAUNCHER"
    chmod +x "$LAUNCHER"

    echo "installed: $LAUNCHER"
    case ":$PATH:" in
        *":$BIN_DIR:"*) ;;
        *) echo "note: $BIN_DIR is not on PATH; add it to your shell profile" ;;
    esac
    echo "run: SHADOW_MODEL=<ollama-model-name> shadow-code"
}

uninstall() {
    local purge="$1"
    if [[ -f "$LAUNCHER" ]]; then
        # Only remove a launcher that points at our venv; never clobber an
        # unrelated shadow-code executable the owner placed there.
        if grep -qF "$VENV_DIR" "$LAUNCHER"; then
            rm -f "$LAUNCHER"
            echo "removed launcher: $LAUNCHER"
        else
            echo "kept foreign launcher (not ours): $LAUNCHER"
        fi
    fi
    if [[ -d "$INSTALL_DIR" ]]; then
        rm -rf "$INSTALL_DIR"
        echo "removed install dir: $INSTALL_DIR"
    fi

    if [[ "$purge" == "1" ]]; then
        echo "about to delete user data:"
        echo "  $STATE_DIR   (sessions, events, prompt snapshots, backups)"
        echo "  $CONFIG_DIR  (prompt overlays)"
        local answer
        read -r -p "type 'purge' to confirm deletion: " answer
        if [[ "$answer" == "purge" ]]; then
            rm -rf "$STATE_DIR" "$CONFIG_DIR"
            echo "purged user data"
        else
            echo "purge aborted; user data preserved"
        fi
    else
        echo "user data preserved:"
        echo "  $STATE_DIR"
        echo "  $CONFIG_DIR"
        echo "re-run with 'uninstall --purge' to delete it"
    fi
}

mode="install"
purge="0"
for arg in "$@"; do
    case "$arg" in
        install | update) mode="install" ;;
        uninstall | remove) mode="uninstall" ;;
        --purge) purge="1" ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            die "unknown argument: $arg"
            ;;
    esac
done

if [[ "$mode" == "install" && "$purge" == "1" ]]; then
    die "--purge only applies to uninstall"
fi

case "$mode" in
    install) install ;;
    uninstall) uninstall "$purge" ;;
esac
