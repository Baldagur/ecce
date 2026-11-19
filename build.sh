#!/usr/bin/env bash
# Build a single double-click executable: dist/psdprep
# Needs (offline is fine): python3 -m pip install --no-index --find-links wheels -r requirements.txt pyinstaller
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
EXTRA=()
if compgen -G "profiles/*.ic[cm]" > /dev/null; then
  EXTRA+=(--add-data "profiles:profiles")
fi
"$PY" -m PyInstaller --noconfirm --clean --onefile --windowed --name psdprep \
  --exclude-module tkinter \
  "${EXTRA[@]}" main.py
echo "Built: $(pwd)/dist/psdprep"
