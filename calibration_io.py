"""Read ChArUco results directly in the xArm base and color optical frames."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import yaml


def transform_points(points_xyz, matrix):
    points = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    return points @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


@dataclass
class CameraCalibration:
    serial: str
    transform: np.ndarray
    color_K: np.ndarray
    color_D: np.ndarray

    @classmethod
    def read(cls, path):
        data = json.loads(Path(path).expanduser().read_text())
        if data.get('units') != 'meters':
            raise ValueError(f'Calibration must use meters: {path}')
        matrix = np.asarray(data['T_base_color'], dtype=float)
        K = np.asarray(data['color_K'], dtype=float)
        D = np.asarray(data['color_D'], dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError('Invalid T_base_color')
        if not np.allclose(matrix[3], [0, 0, 0, 1]) or not np.allclose(matrix[:3,:3].T @ matrix[:3,:3], np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(matrix[:3,:3]), 1, atol=1e-4):
            raise ValueError('T_base_color must be a rigid transform')
        if K.shape != (3,3) or not np.isfinite(K).all() or min(K[0,0], K[1,1]) <= 0 or not np.isfinite(D).all():
            raise ValueError('Invalid color intrinsics')
        # The ported depth lifter uses pinhole deprojection. These calibrated streams have D=0.
        if np.any(np.abs(D) > 1e-9):
            raise ValueError('Nonzero distortion needs rectification before pinhole lifting')
        return cls(str(data['serial']), matrix, K, D)


class TransformChain:
    def __init__(self, cameras, tolerance_px=0.5):
        self.cameras = cameras
        self.tolerance_px = float(tolerance_px)
        self.t_base_cam0 = cameras[0].transform
        self.t_base_cam1 = cameras[1].transform
        self.t_cam0_cam1 = np.linalg.inv(self.t_base_cam0) @ self.t_base_cam1

    def transform_points_camera_to_base(self, camera_id, points_xyz):
        return transform_points(points_xyz, self.cameras[int(camera_id)].transform)

    def validate_frame(self, camera_id, bundle):
        calibration = self.cameras[int(camera_id)]
        if bundle.serial != calibration.serial:
            raise ValueError(f'cam{camera_id}: calibration serial mismatch')
        K = calibration.color_K
        expected = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
        actual = np.array([bundle.intrinsics[k] for k in ('fx','fy','cx','cy')])
        if not np.isfinite(actual).all() or np.max(np.abs(actual - expected)) > self.tolerance_px:
            raise ValueError(f'cam{camera_id}: active color intrinsics differ from calibration; use the calibrated stream profile')
        if bundle.color_image.shape[:2] != bundle.depth_image_m.shape:
            raise ValueError('Depth must be aligned to color')
        return dict(zip(('fx','fy','cx','cy'), expected.tolist()))


def load_transform_chain(config_path='configs/handover.yaml'):
    path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(path.read_text())
    paths = config['calibration']['cameras']
    cameras = []
    for i in range(2):
        candidate = Path(paths[f'cam{i}']).expanduser()
        if not candidate.is_absolute(): candidate = path.parent / candidate
        camera = CameraCalibration.read(candidate)
        if str(config['cameras'][f'cam{i}']['serial']) != camera.serial:
            raise ValueError(f'cam{i}: configured serial differs from calibration')
        cameras.append(camera)
    if cameras[0].serial == cameras[1].serial:
        raise ValueError('Two distinct camera serials are required')
    return TransformChain(cameras, config['calibration'].get('intrinsics_tolerance_px',0.5))
