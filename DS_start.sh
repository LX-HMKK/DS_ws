#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMMAND="
sleep 10
cd \"$SCRIPT_DIR\"
python3 scripts/main.py
"

gnome-terminal -- bash -c "$COMMAND; exec bash"
