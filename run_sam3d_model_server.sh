#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SAM3D_PYTHON=$(python -c 'import yaml; print(yaml.safe_load(open("configs/handover.yaml"))["runtime"]["sam3d_server"]["python_interpreter"])')
SAM3D_REPO=$(python -c 'import yaml; print(yaml.safe_load(open("configs/handover.yaml"))["runtime"]["sam3d_server"]["repo_path"])')
SAM3D_SOCKET=$(python -c 'import yaml; print(yaml.safe_load(open("configs/handover.yaml"))["runtime"]["sam3d_server"]["server_socket"])')
exec "$SAM3D_PYTHON" tools/sam3d_model_server.py --config configs/handover.yaml --sam3d-repo "$SAM3D_REPO" --socket "$SAM3D_SOCKET" "$@"
