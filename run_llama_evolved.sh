#!/bin/zsh
# Run llama-alpha, llama-beta, llama-gamma to 100 sessions each.

DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$DIR/.venv/bin/python3"

run_to_100() {
    local instance=$1
    while true; do
        count=$(python3 -c "import json; d=json.load(open('$DIR/instances/$instance/.dmn_state.json')); print(d['session_count'])" 2>/dev/null || echo 0)
        if [ "$count" -ge 100 ]; then
            echo "$(date) — [$instance] reached 100 sessions, stopping."
            break
        fi
        echo "$(date) — [$instance] session $count"
        output=$("$PYTHON" "$DIR/dmn.py" --instance "$instance" 2>&1)
        if echo "$output" | grep -q "rate_limit_exceeded\|Rate limit\|429"; then
            wait_mins=$(echo "$output" | grep -oE 'try again in [0-9]+m' | grep -oE '[0-9]+' | head -1)
            wait_mins=${wait_mins:-35}
            echo "$(date) — [$instance] rate limit hit, waiting ${wait_mins}m..."
            sleep $(( wait_mins * 60 + 60 ))
        else
            echo "$output"
            sleep 2
        fi
    done
}

run_to_100 llama-alpha
run_to_100 llama-beta
run_to_100 llama-gamma

echo "$(date) — All llama evolved instances complete."
