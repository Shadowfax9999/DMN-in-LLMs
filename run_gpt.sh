#!/bin/zsh
# Run all 5 GPT-4o instances in parallel.
# Each produces one session; evolution triggers automatically every 5th.

DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/.venv/bin/python3"
LOG="$DIR/gpt_instances.log"

echo "$(date) — Running GPT instances (parallel)..." >> "$LOG"

for instance in gpt-null gpt-alpha gpt-beta gpt-gamma gpt-perturb; do
    (
        echo "$(date) — [$instance] starting" >> "$LOG"
        "$PYTHON" "$DIR/dmn.py" --instance "$instance" >> "$LOG" 2>&1
        echo "$(date) — [$instance] done" >> "$LOG"
    ) &
done

wait

echo "$(date) — All GPT instances complete." >> "$LOG"
