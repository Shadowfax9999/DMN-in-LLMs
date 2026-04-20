#!/bin/zsh
# Run all 4 Llama DMN instances in parallel via Groq.
# Each produces one session; evolution triggers automatically every 5th.
# NOTE: Groq free tier = ~100k tokens/day. Run in batches of ~30 sessions max.

DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/.venv/bin/python3"
LOG="$DIR/llama_instances.log"

echo "$(date) — Running Llama instances (parallel)..." >> "$LOG"

for instance in llama-alpha llama-beta llama-gamma llama-perturb; do
    (
        echo "$(date) — [$instance] starting" >> "$LOG"
        "$PYTHON" "$DIR/dmn.py" --instance "$instance" >> "$LOG" 2>&1
        echo "$(date) — [$instance] done" >> "$LOG"
    ) &
done

wait

echo "$(date) — All Llama instances complete." >> "$LOG"
