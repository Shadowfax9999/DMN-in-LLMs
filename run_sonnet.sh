#!/bin/zsh
# Run all Sonnet DMN instances in parallel.

DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/.venv/bin/python3"
LOG="$DIR/sonnet.log"

echo "$(date) — Running Sonnet instances (parallel)..." >> "$LOG"

for instance in sonnet-alpha sonnet-beta sonnet-gamma sonnet-perturb; do
    (
        echo "$(date) — [$instance] starting" >> "$LOG"
        "$PYTHON" "$DIR/dmn.py" --instance "$instance" >> "$LOG" 2>&1
        echo "$(date) — [$instance] done" >> "$LOG"
    ) &
done

wait

echo "$(date) — All Sonnet instances complete." >> "$LOG"
