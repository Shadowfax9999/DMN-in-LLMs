#!/bin/zsh
# One-off runs for today, then self-deletes the launchd jobs

export PATH="$HOME/.local/bin:$PATH"
DIR="/Users/charliemurray/Documents/creativity work"

cd "$DIR"
/bin/zsh "$DIR/run_dmn.sh"
