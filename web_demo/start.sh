#!/bin/bash
# Startup script for the web demo (uses current conda environment)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting Real-time 3D Motion Generation Demo..."
echo "Using current Python: $(which python)"
echo ""
echo "Available pages:"
echo "  http://localhost:5000      - Standard audio capture mode"
echo "  http://localhost:5000/mixer - Live mixer mode (multi-track + speaker)"
echo ""

python app.py
