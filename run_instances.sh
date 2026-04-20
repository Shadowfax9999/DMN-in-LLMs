#!/bin/zsh
# Run all 3 DMN instances in parallel.
# Each produces one session; evolution triggers automatically every 5th.

DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/.venv/bin/python3"
LOG="$DIR/instances.log"

echo "$(date) — Running instances (parallel)..." >> "$LOG"

for instance in alpha beta gamma; do
    (
        echo "$(date) — [$instance] starting" >> "$LOG"
        "$PYTHON" "$DIR/dmn.py" --instance "$instance" >> "$LOG" 2>&1
        echo "$(date) — [$instance] done" >> "$LOG"
    ) &
done

wait

echo "$(date) — All instances complete." >> "$LOG"
