from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Any
import cv2 as cv
import numpy as np
import yaml

def _environment_for(interpreter: Path) -> dict[str, str]:
    environment = dict(os.environ)
    env_root = interpreter.resolve().parents[1]
    environment['CONDA_PREFIX'] = str(env_root)
    environment['LIDRA_SKIP_INIT'] = 'true'
    env_lib = env_root / 'lib'
    if env_lib.is_dir():
        existing = environment.get('LD_LIBRARY_PATH', '')
        environment['LD_LIBRARY_PATH'] = f'{env_lib}:{existing}' if existing else str(env_lib)
    return environment

def generate_runtime_template(*, config_path: str | Path, artifact_dir: str | Path, repo_root: str | Path, execution_mode: str, sam3d_python: str | Path, sam3d_repo: str | Path, server_socket: str, server_ready_timeout_s: float, request_timeout_s: float) -> Path:
    """Generate ``template.npy`` from already-written runtime artifacts."""
    config_path = Path(config_path).expanduser().resolve()
    artifact_dir = Path(artifact_dir).expanduser().resolve()
    repo_root = Path(repo_root).expanduser().resolve()
    sam3d_python = Path(sam3d_python).expanduser().resolve()
    sam3d_repo = Path(sam3d_repo).expanduser()
    if not sam3d_repo.is_absolute():
        sam3d_repo = repo_root / sam3d_repo
    sam3d_repo = sam3d_repo.resolve()
    mode = str(execution_mode).strip().lower()
    if mode not in {'server', 'oneshot'}:
        raise ValueError(f'Unsupported SAM3D execution mode: {execution_mode!r}')
    if mode == 'server':
        from tools.sam3d_ipc import build_model_spec, generate, wait_until_ready
        model_spec = build_model_spec(config_path, sam3d_repo)
        wait_until_ready(server_socket, expected_signature=model_spec['signature'], ready_timeout_s=float(server_ready_timeout_s))
        generate(server_socket, config_path=config_path, artifact_dir=artifact_dir, expected_signature=model_spec['signature'], timeout_s=float(request_timeout_s))
    else:
        command = [str(sam3d_python), str(repo_root / 'tools' / 'sam3d_generate.py'), '--config', str(config_path), '--artifact-dir', str(artifact_dir), '--repo-root', str(repo_root), '--sam3d-repo', str(sam3d_repo)]
        result = subprocess.run(command, cwd=str(repo_root), env=_environment_for(sam3d_python), capture_output=True, text=True, timeout=float(request_timeout_s), check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or 'unknown error').strip()
            raise RuntimeError(f'SAM3D oneshot generation failed with exit code {result.returncode}: {detail}')
    template_path = artifact_dir / 'template.npy'
    if not template_path.is_file():
        raise RuntimeError(f'SAM3D generation completed without template: {template_path}')
    return template_path

def validate_template_points(template_path: str | Path, *, min_template_points: int) -> np.ndarray:
    """Load and validate a generated point cloud before tracker replacement."""
    path = Path(template_path).expanduser().resolve()
    points = np.load(path, allow_pickle=False)
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f'SAM3D template must have shape (N,3), got {points.shape}')
    if len(points) < max(1, int(min_template_points)):
        raise ValueError(f'SAM3D template has {len(points)} points; minimum is {int(min_template_points)}')
    if not np.isfinite(points).all():
        raise ValueError('SAM3D template contains non-finite coordinates')
    return points
