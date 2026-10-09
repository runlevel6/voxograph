#!/bin/bash

# Script to activate the venv and run voxograph-ui.
# Safe to launch from anywhere (desktop shortcut, file manager, terminal).

# Resolve paths from this script's location so the CWD does not matter.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="/home/vlad/audiblez_venv"

# Activate the virtual environment.
source "$VENV_DIR/bin/activate"

# Check if the virtual environment was activated successfully
if [ $? -ne 0 ]; then
    echo "Error: Could not activate the virtual environment."
    echo "Please ensure that '$VENV_DIR/bin/activate' exists and is correct."
    exit 1
fi

# Run the GUI from the project directory so relative assets resolve.
cd "$SCRIPT_DIR" || exit 1
echo "Activating virtual environment and starting voxograph-ui..."
voxograph-ui

# Optional: Add a message for when voxograph-ui finishes or exits
echo "voxograph-ui session ended."

exit 0
