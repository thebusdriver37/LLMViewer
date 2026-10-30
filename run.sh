#!/bin/zsh
# Start LLM Viewer at http://127.0.0.1:8765  (override the model with LLMVIEW_MODEL=...)
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
fi
PYTHONUNBUFFERED=1 exec .venv/bin/python server.py
