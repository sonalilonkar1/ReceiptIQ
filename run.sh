#!/bin/bash
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd):$PYTHONPATH"
./.venv/bin/python -m app.main
