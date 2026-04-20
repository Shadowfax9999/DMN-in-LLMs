#!/bin/zsh
# DMN runner — called by launchd twice daily
# Uses Claude Code CLI (no API key needed)
#
# Runs a generation session every time.
# Runs evolution every EVOLVE_EVERY sessions (default: 5).

# Ensure claude CLI is on PATH
export PATH="$HOME/.local/bin:$PATH"

DIR="/Users/charliemurray/Documents/creativity work"
PYTHON="$DIR/.venv/bin/python3"
LOG="$DIR/dmn.log"

cd "$DIR"

# ── Generate ───────────────────────────────────────────────────────────────────
"$PYTHON" "$DIR/dmn.py" >> "$LOG" 2>&1

# ── Evolve periodically ───────────────────────────────────────────────────────
EVOLVE_EVERY=${EVOLVE_EVERY:-5}
SESSION_COUNT=$(python3 -c "import json; print(json.load(open('.dmn_state.json'))['session_count'])" 2>/dev/null || echo 0)

if (( SESSION_COUNT % EVOLVE_EVERY == 0 )) && (( SESSION_COUNT > 0 )); then
    echo "$(date): Triggering evolution after session #$SESSION_COUNT" >> "$LOG"
    "$PYTHON" "$DIR/evolve.py" >> "$LOG" 2>&1
fi
