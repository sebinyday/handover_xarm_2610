#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec python handover.py --config configs/handover.yaml "$@"
