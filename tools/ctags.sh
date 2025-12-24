#!/usr/bin/env bash
set -euo pipefail

# Go to repo root (works even if run from subdir)
ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

ctags -R \
  --languages=Python \
  --python-kinds=-iv \
  --fields=+iaS \
  --extras=+q \
  --exclude=.git \
  --exclude=.venv \
  --exclude=venv \
  --exclude=__pycache__ \
  --exclude=ride_files \
  --exclude=*.pyc \
  -f .tags \
  .

