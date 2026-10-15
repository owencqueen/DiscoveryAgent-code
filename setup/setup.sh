#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

if ! command -v uv >/dev/null 2>&1; then
    echo "Install uv from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

uv sync --locked "$@"
.venv/bin/python -c "import DiscoveryAgent; from DiscoveryAgent.hypothesis_brancher import HypothesisBrancher; print('DiscoveryAgent import OK')"
