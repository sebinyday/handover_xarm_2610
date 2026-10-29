"""Dual-camera grasp-target follow script adapted for xArm7 xArm control."""
import argparse
import os
import queue
import sys
import time
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from collections import deque
if __package__ is None or __package__ == '':
    sys.path.append(str(Path(__file__).resolve().parent))
import cv2 as cv
import numpy as np
import yaml
from calibration_io import load_transform_chain
from perception.fusion import PerceptionFusion
from perception.grasp_target import GraspTargetPlanner
from perception.grasp_z_stabilizer import GraspPointZStabilizer
from perception.hand_relative_fallback import HandRelativeFallbackTracker
from perception.hand_selector import HandSelector
from perception.hand_worker import HandWorkerCam0, HandWorkerCam1
from perception.object_merger import ObjectMerger
from perception.silhouette_constraint import SilhouetteObservation
from perception.shape_fitting_tracker_v2 import ShapeFittingTracker
from pipeline import build_dual_perception_pipeline
from perception.sam3d_live_runtime import Sam3DLiveRuntime, enforce_sam3d_target_gate
from perception.target_predictor import TargetPredictor
from robot.xarm_controller import XArmController
from system.dual_sensor_hub import DualSensorHub
from system.shared_state import GRIPPER_CLOSE, GRIPPER_HOLD, GRIPPER_OPEN, ROBOT_CMD_HOLD, ROBOT_CMD_MOVE_TO_POSITION, ROBOT_CMD_SERVO_TO_POSITION, ROBOT_CMD_STOP, RobotCommandState
from video_record import HandoverVideoRecorderService
from utils.debug_3d_recorder import Debug3DRecorder
from utils.handover_metadata import HandoverMetadataRecorder
from utils.runtime_profiler import RuntimeProfiler
DEFAULT_CONFIG_PATH = Path('configs/handover.yaml')
ABLATION_BASELINE = 'baseline'
ABLATION_SHAPE_FITTING = 'shape-fitting'
ABLATION_TACTILE_SENSING = 'tactile-sensing'
ABLATION_SILHOUETTE_SCALING = 'silhouette-scaling'
ABLATION_CHOICES = (ABLATION_BASELINE, ABLATION_SHAPE_FITTING, ABLATION_TACTILE_SENSING, ABLATION_SILHOUETTE_SCALING)
GRASP_DETECTION_HYBRID = 'hybrid'
GRASP_DETECTION_ROBOTIQ_OBJECT_ONLY = 'robotiq_object_only'
READINESS_SILHOUETTE = 'silhouette_ready'
READINESS_SHAPE_FIT = 'shape_fit_ready'
READINESS_RAW_CLOUD = 'raw_cloud_ready'
REFERENCE_LOCK_COUNT = 8
MOTION_TRIGGER_MM = 50.0
MOTION_TRIGGER_Z_MM = 50.0
DEFAULT_CONTROL_HZ = 30.0
MAX_XY_SPEED_MM_S = 250.0
MAX_Z_SPEED_MM_S = 250.0
EEF_X_OFFSET_MM = 0.0
EEF_Y_OFFSET_MM = 0.0
EEF_Z_OFFSET_MM = 0.0
HOVER_Z_OFFSET_MM = 0
DESCEND_EXTRA_MM = 0.0
BACKOFF_X_MM = 0.0
DEFAULT_POST_RELEASE_Z_OFFSET_MM = 0.0
DEFAULT_PLACE_OBJECT_XY_MM = (0.0, 0.0)
HOME_PLACE_MIN_Z_MM = 40.0
PRE_RELEASE_MIN_Z_EPSILON_MM = 0.001
GRASP_POINT_Y_OFFSET_MM = 0.0
GRASP_POINT_Z_OFFSET_MM = 10.0
PLACE_Z_GRASP_BUFFER_FRAMES = 5
PLACE_Z_MIN_VALID_SAMPLES = 3
GRIPPER_FORCE_STOP_DELTA_N = 100000
GRIPPER_FORCE_STOP_MIN_ELAPSED_S = 0.12
ROBOTIQ_OBJ_COMMAND_GRACE_S = 0.15
ROBOTIQ_OBJ_POSITION_START_DELTA = 2
DEFAULT_GRIPPER_POSITION_COMPLETE_THRESHOLD = 200
DEFAULT_GRIPPER_POSITION_STALL_ENABLED = True
DEFAULT_GRIPPER_POSITION_STALL_STABLE_READS = 3
DEFAULT_GRIPPER_POSITION_STALL_TOLERANCE = 1
DEFAULT_GRIPPER_POSITION_STALL_MIN_ELAPSED_S = 0.12
RELEASE_PARAMETER_MM = 80.0
TACTILE_SENSING_ABLATION_RELEASE_PARAMETER_MM = 20.0
DEFAULT_TACTILE_PORT = '/dev/ttyACM0'
DEFAULT_TACTILE_NUM_MAGS = 5
DEFAULT_TACTILE_BASELINE_SAMPLES = 5
DEFAULT_TACTILE_STARTUP_DELAY_S = 1.0
DEFAULT_TACTILE_CONTACT_NORM_THRESHOLD = 30.0
DEFAULT_TACTILE_EXTRA_GRASP_POS = 10
DEFAULT_TACTILE_RELEASE_DELTA_THRESHOLD = 50.0
DEFAULT_TACTILE_RELEASE_REF_DELAY_S = 0.3
DEFAULT_TACTILE_RELEASE_TIMEOUT_S = 1.0
DEFAULT_TACTILE_RELEASE_DESCENT_STEP_MM = 2.0
DEFAULT_TACTILE_RELEASE_DESCENT_POLL_DT_S = 0.03
DEFAULT_TACTILE_AUTO_BASELINE_RESET_AFTER_OPEN_S = 1.5
DEFAULT_TACTILE_RELEASE_STOP_TIMING_DEBUG = False
DEFAULT_TACTILE_RELEASE_STOP_SPEED_THRESHOLD_MPS = 0.002
DEFAULT_TACTILE_RELEASE_STOP_MONITOR_TIMEOUT_S = 1.0
DEFAULT_TACTILE_RELEASE_STOP_MONITOR_POLL_DT_S = 0.005
DEFAULT_POST_BACKOFF_STOP_CHECK_ENABLED = True
DEFAULT_POST_BACKOFF_STOP_SPEED_THRESHOLD_MPS = 0.002
DEFAULT_POST_BACKOFF_STOP_TIMEOUT_S = 1.0
DEFAULT_POST_BACKOFF_STOP_POLL_DT_S = 0.01
DEFAULT_POST_BACKOFF_STOP_REQUIRE_CONFIRMED = False
HOME_JOINTS_DEG = []
HOME_JOINT_TOLERANCE_DEG = 1.0
HOME_JOINT_SPEED_RAD_S = 0.5
HOME_JOINT_ACCELERATION_RAD_S2 = 0.5
VIDEO_RECORDER_SERIAL = '335522072904'
VIDEO_RECORDER_PORT = 5000
ROBOT_REQ_INIT_ROBOT = 'INIT_ROBOT'
ROBOT_REQ_START_FOLLOW = 'START_FOLLOW'
ROBOT_REQ_STOP_FOLLOW = 'STOP_FOLLOW'
ROBOT_REQ_START_GRASP_PLACE = 'START_GRASP_PLACE'
ROBOT_REQ_RESET_HOME = 'RESET_HOME'
ROBOT_REQ_SAVE_AND_STOP = 'SAVE_AND_STOP'
ROBOT_REQ_EMERGENCY_STOP = 'EMERGENCY_STOP'
ROBOT_REQ_SHUTDOWN = 'SHUTDOWN'
ROBOT_URGENT_REQUESTS = {ROBOT_REQ_RESET_HOME, ROBOT_REQ_SAVE_AND_STOP, ROBOT_REQ_EMERGENCY_STOP, ROBOT_REQ_SHUTDOWN}
ROBOT_STATE_IDLE = 'IDLE'
ROBOT_STATE_INITIALIZING = 'INITIALIZING'
ROBOT_STATE_FOLLOWING = 'FOLLOWING'
ROBOT_STATE_GRASPING = 'GRASPING'
ROBOT_STATE_RETURNING = 'RETURNING'
ROBOT_STATE_PLACING = 'PLACING'
ROBOT_STATE_RESETTING = 'RESETTING'
ROBOT_STATE_DONE = 'DONE'
ROBOT_STATE_ERROR = 'ERROR'
ROBOT_STATE_STOPPING = 'STOPPING'

@dataclass
class RobotRequest:
    type: str
    payload: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    request_id: int = 0

@dataclass
class RobotActionContext:
    fitted_points_base: object = None
    grasp_point_base: object = None
    object_label: object = None
    template_axes_base: object = None

@dataclass
class RobotStatus:
    state: str = ROBOT_STATE_IDLE
    last_error: object = None
    is_connected: bool = False
    using_mock: bool = False
    active_request: object = None
    active_request_id: int = 0
    last_robot_pose: object = None
    last_command_type: object = None
    grasp_ok: object = None
    task_done_epoch: int = 0
    reset_done_epoch: int = 0
    task_ready_epoch: int = 0
    home_ready: bool = False

def parse_args():
    """Parse CLI switches for perception, robot follow, debugging, and profiling."""
    parser = argparse.ArgumentParser(description='Run dual-camera perception, build a grasp target from hand pose + merged object cloud, and follow it with xArm7 xArm.')
    parser.add_argument('--width', type=int, default=None, help='Color/depth stream width.')
    parser.add_argument('--height', type=int, default=None, help='Color/depth stream height.')
    parser.add_argument('--fps', type=int, default=None, help='RealSense stream FPS.')
    parser.add_argument('--depth-max-m', type=float, default=1.5, help='Upper bound for depth visualization.')
    parser.add_argument('--config', default=str(DEFAULT_CONFIG_PATH), help='YAML config used by the existing xArm7 xArm controller.')
    parser.add_argument('--enable-follow', action='store_true', help='Enable xArm7 xArm follow mode.')
    parser.add_argument('--robot-ip', type=str, default=None, help='Optional xArm7 IP override.')
    parser.add_argument('--min-valid-count', type=int, default=3, help='Min consecutive valid detections before follow.')
    parser.add_argument('--target-timeout-s', type=float, default=0.5, help='Stop following if target is stale.')
    parser.add_argument('--move-to-base', action='store_true', help='Reserved for compatibility; HOME now uses joint targets.')
    parser.add_argument('--open-gripper', action='store_true', help='Open gripper during init.')
    parser.add_argument('--verbose-robot', action='store_true', help='Print detailed robot command logs.')
    parser.add_argument('--follow-z', dest='follow_z', action='store_true', help='Enable z-axis follow.')
    parser.add_argument('--no-follow-z', dest='follow_z', action='store_false', help='Disable z-axis follow.')
    parser.set_defaults(follow_z=None)
    parser.add_argument('--control-hz', type=float, default=None, help='Servo loop rate. Defaults to config or 30 Hz.')
    parser.add_argument('--position-tolerance-m', type=float, default=0.01, help='Move completion tolerance in meters.')
    parser.add_argument('--move-timeout-s', type=float, default=10.0, help='Timeout for blocking move steps.')
    parser.add_argument('--gripper-close-timeout-s', type=float, default=None, help='Gripper motion/verification timeout; defaults to robot.gripper.close_timeout_sec.')
    parser.add_argument('--gripper-release-dwell-s', type=float, default=0.5, help='Dwell after opening gripper.')
    parser.add_argument('--apply-place-grasp-offset-xy', dest='apply_place_grasp_offset_xy', action='store_true', help='Apply the grasp-time object-to-TCP XY offset to the configured place-object XY target.')
    parser.add_argument('--disable-place-grasp-offset-xy', dest='apply_place_grasp_offset_xy', action='store_false', help='Use place_object_xy_mm directly as the final TCP XY without grasp-offset compensation.')
    parser.set_defaults(apply_place_grasp_offset_xy=None)
    parser.add_argument('--enable-pre-release-descend-before-open', dest='pre_release_descend_before_open', action='store_true', help='Descend in Z before opening the gripper at PLACE. Defaults to config.')
    parser.add_argument('--pre-release-descend-m', type=float, default=None, help='Optional pre-release descend distance in meters. Defaults to RELEASE_PARAMETER_MM.')
    parser.set_defaults(pre_release_descend_before_open=None)
    parser.add_argument('--follow-handoff-timeout-s', type=float, default=5.0, help='Timeout to wait for the follow thread to release robot control before pregrasp.')
    parser.add_argument('--enable-target-prediction', dest='enable_target_prediction', action='store_true', help='Use Kalman prediction to bridge short target dropouts during FOLLOW.')
    parser.add_argument('--disable-target-prediction', dest='enable_target_prediction', action='store_false', help='Disable target prediction and fall back to raw target only.')
    parser.set_defaults(enable_target_prediction=True)
    parser.add_argument('--prediction-max-horizon-s', type=float, default=0.25, help='Maximum dropout duration bridged by predicted targets.')
    parser.add_argument('--prediction-process-noise-mm-s2', type=float, default=800.0, help='Kalman process noise acceleration scale for xy prediction.')
    parser.add_argument('--prediction-measurement-noise-mm', type=float, default=25.0, help='Kalman measurement noise scale for xy updates.')
    parser.add_argument('--prediction-max-xy-speed-mm-s', type=float, default=200.0, help='Maximum xy prediction speed before velocity clipping.')
    parser.add_argument('--prediction-reinit-jump-mm', type=float, default=120.0, help='Reinitialize the Kalman state if a new measurement jumps too far from prediction.')
    parser.add_argument('--3d-debug', dest='debug_3d', action='store_true', help="Enable in-memory 3D debug recording. Press 'd' to save and reset.")
    parser.add_argument('--save-image', action='store_true', help='With --3d-debug, include raw color/depth frames in saved 3D debug recordings.')
    parser.add_argument('--debug-3d-dir', type=str, default='output/debug_3d', help='Directory where saved 3D debug recordings are written.')
    parser.add_argument('--debug-3d-max-object-points', type=int, default=8000, help='Maximum raw object cloud points stored per debug frame.')
    parser.add_argument('--debug-3d-max-template-points', type=int, default=8000, help='Maximum fitted template cloud points stored per debug frame.')
    parser.add_argument('--no-debug-3d-template-axes', dest='debug_3d_template_axes', action='store_false', help='Do not store fitted template local x/y/z axes in 3D debug recordings.')
    parser.set_defaults(debug_3d_template_axes=True)
    parser.add_argument('--disable-debug-3d-recording', action='store_true', help='Backward-compatible alias that keeps 3D debug recording disabled.')
    parser.add_argument('--record-video', action='store_true', help="Enable task video recording. Press 's' to finish/save using the configured recorder flow.")
    parser.add_argument('--profile-runtime', action='store_true', help="Collect runtime CPU/RAM/GPU and per-stage timing logs. Press 's' to save the current session.")
    parser.add_argument('--profile-dir', type=str, default='output/runtime_profile', help='Directory where runtime profiling CSV/JSON files are written.')
    parser.add_argument('--profile-sample-interval-s', type=float, default=1.0, help='Resource sampling interval for runtime profiling.')
    parser.add_argument('--profile-print-every-s', type=float, default=5.0, help='Print a short runtime profiling summary every N seconds. Use 0 to disable.')
    parser.set_defaults(ablation='baseline', object_backend='sam3d', model='SAM2.1-small', prompt=None, serial=None, imgsz=1024, device='cuda', half=False, select_mode='single_object', select_class=None)
    return parser.parse_args()

def load_yaml_config(path):
    """Load the YAML runtime config used by cameras, perception, safety, and xArm."""
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}

def get_video_recorder_web_ui_enabled(config):
    """Return whether task video recording should use the web UI finalize flow."""
    video_cfg = dict((config or {}).get('video_recording', {}) or {})
    web_ui_cfg = dict(video_cfg.get('web_ui', {}) or {})
    return bool(web_ui_cfg.get('enabled', True))

def get_video_recorder_tactile_logging_config(config):
    """Return tactile sidecar logging switches for the task video recorder."""
    video_cfg = dict((config or {}).get('video_recording', {}) or {})
    tactile_cfg = dict(video_cfg.get('tactile_logging', {}) or {})
    return {'enabled': bool(tactile_cfg.get('enabled', True)), 'csv_enabled': bool(tactile_cfg.get('csv_enabled', True)), 'rerun_enabled': bool(tactile_cfg.get('rerun_enabled', True)), 'rerun_live': bool(tactile_cfg.get('rerun_live', False))}

def apply_config_defaults(args, config):
    configure_teaching(args, config)
    for name in ('width', 'height', 'fps'):
        value = getattr(args, name)
        configured = int(config['cameras']['cam0'][name])
        if value is not None and value != configured:
            raise ValueError(f'--{name} must match the calibrated camera configuration ({configured})')
        setattr(args, name, configured)
    'Fill CLI defaults from config while keeping explicit command-line values.'
    args.ablation = str(getattr(args, 'ablation', ABLATION_BASELINE)).strip().lower()
    if 'baseline' not in ABLATION_CHOICES:
        raise ValueError(f"Unsupported ablation mode={'baseline'!r}; expected one of {ABLATION_CHOICES}")
    robot_cfg = config.get('robot', {})
    live_cfg = robot_cfg.get('live_follow', {})
    return_sequence_cfg = robot_cfg.get('return_sequence', {})
    tactile_cfg = robot_cfg.get('tactile', {})
    xarm_cfg = robot_cfg.get('xarm', {})
    safety_cfg = config.get('safety', {})
    workspace_cfg = safety_cfg.get('workspace_bounds_m', {})
    args.shape_fitting_ablation = 'baseline' == ABLATION_SHAPE_FITTING
    args.silhouette_scaling_ablation = 'baseline' == ABLATION_SILHOUETTE_SCALING
    args.grasp_detection_mode = GRASP_DETECTION_ROBOTIQ_OBJECT_ONLY if 'baseline' == ABLATION_TACTILE_SENSING else GRASP_DETECTION_HYBRID
    args.release_parameter_mm = TACTILE_SENSING_ABLATION_RELEASE_PARAMETER_MM if 'baseline' == ABLATION_TACTILE_SENSING else RELEASE_PARAMETER_MM
    if args.robot_ip is None:
        args.robot_ip = xarm_cfg.get('robot_ip')
    if args.control_hz is None:
        args.control_hz = float(live_cfg.get('control_hz', DEFAULT_CONTROL_HZ))
    if args.follow_z is None:
        args.follow_z = bool(live_cfg.get('follow_z', False))
    if args.gripper_close_timeout_s is None:
        args.gripper_close_timeout_s = float(robot_cfg.get('gripper',{}).get('close_timeout_sec',5.0))
    if not np.isfinite(args.gripper_close_timeout_s) or args.gripper_close_timeout_s<=0:
        raise ValueError('Gripper close timeout must be positive and finite')
    place_object_xy_mm = return_sequence_cfg.get('place_object_xy_mm', DEFAULT_PLACE_OBJECT_XY_MM)
    if not isinstance(place_object_xy_mm, (list, tuple)) or len(place_object_xy_mm) != 2:
        raise ValueError('robot.return_sequence.place_object_xy_mm must contain exactly [x_mm, y_mm]')
    args.place_object_xy_mm = tuple((float(value) for value in place_object_xy_mm))
    if not np.all(np.isfinite(np.asarray(args.place_object_xy_mm, dtype=np.float64))):
        raise ValueError('robot.return_sequence.place_object_xy_mm must contain finite values')
    fixed_xyz = return_sequence_cfg.get('place_object_xyz_mm')
    args.place_object_xyz_mm = None
    args.place_descend_mm = float(return_sequence_cfg.get('place_descend_mm', 0.0))
    if not np.isfinite(args.place_descend_mm) or args.place_descend_mm < 0:
        raise ValueError('place_descend_mm must be finite and nonnegative')
    if fixed_xyz is not None:
        if not isinstance(fixed_xyz, (list, tuple)) or len(fixed_xyz) != 3 or not np.isfinite(fixed_xyz).all():
            raise ValueError('place_object_xyz_mm must contain finite [x_mm, y_mm, z_mm]')
        args.place_object_xyz_mm = tuple(float(v) for v in fixed_xyz)
        for axis, value in zip('xyz', args.place_object_xyz_mm):
            if not workspace_cfg[axis][0] <= value / 1000.0 <= workspace_cfg[axis][1]:
                raise ValueError(f'Fixed place {axis.upper()} outside workspace')
        args.place_object_xy_mm = args.place_object_xyz_mm[:2]
        release_z = (args.place_object_xyz_mm[2] - args.place_descend_mm) / 1000.0
        if not workspace_cfg['z'][0] <= release_z <= workspace_cfg['z'][1]:
            raise ValueError('Place descent target Z outside workspace')
    if getattr(args, 'apply_place_grasp_offset_xy', None) is None:
        args.apply_place_grasp_offset_xy = bool(return_sequence_cfg.get('apply_grasp_offset_xy', True))
    if args.place_object_xyz_mm is not None:
        args.apply_place_grasp_offset_xy = False
    args.post_release_z_offset_mm = float(return_sequence_cfg.get('post_release_z_offset_m', DEFAULT_POST_RELEASE_Z_OFFSET_MM / 1000.0)) * 1000.0
    if args.pre_release_descend_before_open is None:
        args.pre_release_descend_before_open = bool(return_sequence_cfg.get('pre_release_descend_enabled', False))
    pre_release_descend_m = args.pre_release_descend_m if args.pre_release_descend_m is not None else return_sequence_cfg.get('pre_release_descend_m', RELEASE_PARAMETER_MM / 1000.0)
    args.pre_release_descend_mm = max(0.0, float(pre_release_descend_m) * 1000.0)
    args.post_backoff_stop_check_enabled = bool(return_sequence_cfg.get('post_backoff_stop_check_enabled', DEFAULT_POST_BACKOFF_STOP_CHECK_ENABLED))
    args.post_backoff_stop_speed_threshold_mps = max(0.0, float(return_sequence_cfg.get('post_backoff_stop_speed_threshold_mps', DEFAULT_POST_BACKOFF_STOP_SPEED_THRESHOLD_MPS)))
    args.post_backoff_stop_timeout_s = max(0.0, float(return_sequence_cfg.get('post_backoff_stop_timeout_s', DEFAULT_POST_BACKOFF_STOP_TIMEOUT_S)))
    args.post_backoff_stop_poll_dt_s = max(0.0, float(return_sequence_cfg.get('post_backoff_stop_poll_dt_s', DEFAULT_POST_BACKOFF_STOP_POLL_DT_S)))
    args.post_backoff_stop_require_confirmed = bool(return_sequence_cfg.get('post_backoff_stop_require_confirmed', DEFAULT_POST_BACKOFF_STOP_REQUIRE_CONFIRMED))
    for axis_name in ('x', 'y', 'z'):
        arg_name = f'workspace_{axis_name}'
        config_bounds = workspace_cfg.get(axis_name)
        if config_bounds is None or len(config_bounds) != 2:
            raise ValueError(f'Missing safety.workspace_bounds_m.{axis_name} in YAML config.')
        mm_bounds = [float(config_bounds[0]) * 1000.0, float(config_bounds[1]) * 1000.0]
        setattr(args, arg_name, mm_bounds)
    args.tactile_enabled = bool(tactile_cfg.get('enabled', False))
    tolerance=config.get('grasp',{}).get('pregrasp_tolerance_mm',[args.position_tolerance_m*1000]*3)
    if len(tolerance)!=3 or not np.isfinite(tolerance).all() or np.any(np.asarray(tolerance)<=0):
        raise ValueError('grasp.pregrasp_tolerance_mm must contain three positive XYZ tolerances')
    args.pregrasp_tolerance_mm=tuple(float(value) for value in tolerance)
    if 'baseline' == ABLATION_TACTILE_SENSING:
        args.tactile_enabled = False
    args.tactile_port = str(tactile_cfg.get('port', DEFAULT_TACTILE_PORT))
    args.tactile_num_mags = max(1, int(tactile_cfg.get('num_mags', DEFAULT_TACTILE_NUM_MAGS)))
    args.tactile_baseline_samples = max(1, int(tactile_cfg.get('baseline_samples', DEFAULT_TACTILE_BASELINE_SAMPLES)))
    args.tactile_startup_delay_s = max(0.0, float(tactile_cfg.get('startup_delay_s', DEFAULT_TACTILE_STARTUP_DELAY_S)))
    args.tactile_contact_norm_threshold = max(0.0, float(tactile_cfg.get('contact_norm_threshold', DEFAULT_TACTILE_CONTACT_NORM_THRESHOLD)))
    args.tactile_extra_grasp_pos = max(0, int(tactile_cfg.get('extra_grasp_pos', DEFAULT_TACTILE_EXTRA_GRASP_POS)))
    args.tactile_release_delta_threshold = max(0.0, float(tactile_cfg.get('release_delta_threshold', DEFAULT_TACTILE_RELEASE_DELTA_THRESHOLD)))
    args.tactile_release_ref_delay_s = max(0.0, float(tactile_cfg.get('release_ref_delay_s', DEFAULT_TACTILE_RELEASE_REF_DELAY_S)))
    args.tactile_release_descent_min_z_mm = max(HOME_PLACE_MIN_Z_MM, float(tactile_cfg.get('release_descent_min_z_mm', HOME_PLACE_MIN_Z_MM)))
    args.tactile_release_descent_step_mm = max(0.1, float(tactile_cfg.get('release_descent_step_mm', DEFAULT_TACTILE_RELEASE_DESCENT_STEP_MM)))
    args.tactile_release_descent_poll_dt_s = max(0.0, float(tactile_cfg.get('release_descent_poll_dt_s', DEFAULT_TACTILE_RELEASE_DESCENT_POLL_DT_S)))
    args.tactile_auto_baseline_reset_after_open_s = max(0.0, float(tactile_cfg.get('auto_baseline_reset_after_open_s', DEFAULT_TACTILE_AUTO_BASELINE_RESET_AFTER_OPEN_S)))
    args.tactile_release_stop_timing_debug = bool(tactile_cfg.get('release_stop_timing_debug', DEFAULT_TACTILE_RELEASE_STOP_TIMING_DEBUG))
    args.tactile_release_stop_speed_threshold_mps = max(0.0, float(tactile_cfg.get('release_stop_speed_threshold_mps', DEFAULT_TACTILE_RELEASE_STOP_SPEED_THRESHOLD_MPS)))
    args.tactile_release_stop_monitor_timeout_s = max(0.0, float(tactile_cfg.get('release_stop_monitor_timeout_s', DEFAULT_TACTILE_RELEASE_STOP_MONITOR_TIMEOUT_S)))
    args.tactile_release_stop_monitor_poll_dt_s = max(0.0, float(tactile_cfg.get('release_stop_monitor_poll_dt_s', DEFAULT_TACTILE_RELEASE_STOP_MONITOR_POLL_DT_S)))
    args.tactile_debug = bool(tactile_cfg.get('debug', True))
    return args

def render_depth(depth_image_m, max_depth_m):
    max_depth_m = max(max_depth_m, 1e-06)
    clipped = np.clip(depth_image_m, 0.0, max_depth_m)
    scaled = (255.0 * clipped / max_depth_m).astype(np.uint8)
    return cv.applyColorMap(255 - scaled, cv.COLORMAP_TURBO)

class AnySkinTactileManager:
    """Small AnySkin reader used only when robot.tactile.enabled is true."""

    def __init__(self, args):
        self.enabled = bool(getattr(args, 'tactile_enabled', False))
        self.port = str(getattr(args, 'tactile_port', DEFAULT_TACTILE_PORT))
        self.num_mags = max(1, int(getattr(args, 'tactile_num_mags', DEFAULT_TACTILE_NUM_MAGS)))
        self.baseline_samples = max(1, int(getattr(args, 'tactile_baseline_samples', DEFAULT_TACTILE_BASELINE_SAMPLES)))
        self.startup_delay_s = max(0.0, float(getattr(args, 'tactile_startup_delay_s', DEFAULT_TACTILE_STARTUP_DELAY_S)))
        self.contact_norm_threshold = max(0.0, float(getattr(args, 'tactile_contact_norm_threshold', DEFAULT_TACTILE_CONTACT_NORM_THRESHOLD)))
        self.debug = bool(getattr(args, 'tactile_debug', True))
        self.lock = threading.RLock()
        self.stream = None
        self.baseline = None
        self.latest = np.zeros(self.num_mags * 3, dtype=np.float32)
        self.latest_norm = 0.0
        self.last_error = None
        self.status = 'disabled'
        self.release_reference_norm = None
        self.release_delta_norm = None
        self.release_status = 'off'
        self.last_sample_perf_s = np.nan
        self.last_sample_unix_s = np.nan
        self.sample_timeout_s = max(0.25, self.startup_delay_s + 1.0)
        self.last_warn_perf_s = 0.0

    def _warn(self, message, *, min_interval_s=1.0):
        now = time.perf_counter()
        if now - self.last_warn_perf_s >= float(min_interval_s):
            print(message)
            self.last_warn_perf_s = now

    def _stream_is_alive(self):
        if self.stream is None:
            return False
        is_alive = getattr(self.stream, 'is_alive', None)
        if callable(is_alive):
            try:
                return bool(is_alive())
            except Exception:
                return True
        return True

    def _collect_samples(self, num_samples, *, timeout_s=None):
        if self.stream is None:
            return np.empty((0, self.num_mags * 3 + 1), dtype=np.float32)
        num_samples = max(1, int(num_samples))
        timeout_s = self.sample_timeout_s if timeout_s is None else max(0.0, float(timeout_s))
        deadline = time.perf_counter() + timeout_s
        samples = []
        last_cnt = int(getattr(self.stream, 'sample_cnt', 0) or 0)
        if last_cnt > 0:
            samples.append(np.asarray(self.stream.last_reading, dtype=np.float32).copy())
        while len(samples) < num_samples and time.perf_counter() < deadline:
            if not self._stream_is_alive():
                break
            sample_cnt = int(getattr(self.stream, 'sample_cnt', 0) or 0)
            if sample_cnt != last_cnt and sample_cnt > 0:
                samples.append(np.asarray(self.stream.last_reading, dtype=np.float32).copy())
                last_cnt = sample_cnt
                continue
            time.sleep(0.002)
        if not samples:
            return np.empty((0, self.num_mags * 3 + 1), dtype=np.float32)
        return np.asarray(samples, dtype=np.float32)

    def _disable_after_startup_failure(self, reason):
        self.last_error = str(reason)
        self.status = 'unavailable'
        self.release_status = 'off'
        self.enabled = False
        self._warn(f'[WARN] AnySkin tactile reader unavailable on {self.port}; disabling tactile mode. reason={reason}', min_interval_s=0.0)
        self.close()

    def _stream_startup_failure_reason(self, reason):
        port_path = Path(self.port)
        if not port_path.exists():
            return f'{reason}; port does not exist: {self.port}'
        if not os.access(self.port, os.R_OK | os.W_OK):
            return f'{reason}; port is not readable/writable by this user: {self.port}'
        return str(reason)

    def start(self):
        if not self.enabled:
            return
        try:
            from anyskin import AnySkinProcess
        except Exception as exc:
            raise RuntimeError('robot.tactile.enabled is true, but the anyskin package could not be imported. Install AnySkin dependencies or set robot.tactile.enabled: false.') from exc
        try:
            self.stream = AnySkinProcess(num_mags=self.num_mags, port=self.port)
            self.stream.start()
            if hasattr(self.stream, 'start_streaming'):
                self.stream.start_streaming()
            if self.startup_delay_s > 0.0:
                time.sleep(self.startup_delay_s)
            if self._collect_samples(1, timeout_s=self.sample_timeout_s).shape[0] == 0:
                self._disable_after_startup_failure(self._stream_startup_failure_reason('no samples received after starting stream'))
                return
            if not self.reset_baseline():
                self._disable_after_startup_failure(self.last_error or 'baseline reset failed')
                return
            self.status = 'ready'
            if self.debug:
                print(f'[Tactile] AnySkin stream started on {self.port} ({self.num_mags} mags).')
        except Exception as exc:
            self.last_error = str(exc)
            self.status = 'error'
            self.close()
            raise RuntimeError(f'Failed to start AnySkin tactile reader on {self.port}: {exc}') from exc

    def reset_baseline(self):
        with self.lock:
            if self.stream is None:
                return False
            try:
                baseline_data = self._collect_samples(self.baseline_samples)
                if baseline_data.ndim != 2 or baseline_data.shape[1] < self.num_mags * 3 + 1:
                    raise RuntimeError(f'Unexpected AnySkin baseline shape: {baseline_data.shape}')
                self.baseline = np.mean(baseline_data[:, 1:1 + self.num_mags * 3], axis=0)
                self.latest = np.zeros(self.num_mags * 3, dtype=np.float32)
                self.latest_norm = 0.0
                self.last_sample_perf_s = time.perf_counter()
                self.last_sample_unix_s = time.time()
                self.last_error = None
                if self.debug:
                    print('[Tactile] Baseline reset.')
                return True
            except Exception as exc:
                self.last_error = str(exc)
                self.status = 'read_error'
                self._warn(f'[WARN] Tactile baseline reset failed: {exc}')
                return False

    def read(self):
        with self.lock:
            if not self.enabled or self.stream is None:
                return self.latest.copy()
            if self.baseline is None:
                if not self.reset_baseline():
                    return self.latest.copy()
            try:
                sensor_samples = self._collect_samples(1)
                if sensor_samples.shape[0] == 0:
                    raise RuntimeError('No AnySkin samples received')
                sensor_data = sensor_samples[-1]
                sensor_data = np.asarray(sensor_data, dtype=np.float32)[1:1 + self.num_mags * 3]
                if sensor_data.shape[0] != self.num_mags * 3:
                    raise RuntimeError(f'Unexpected AnySkin sample length: {sensor_data.shape[0]}')
                self.latest = sensor_data - self.baseline
                self.latest_norm = float(np.linalg.norm(self.latest))
                self.last_sample_perf_s = time.perf_counter()
                self.last_sample_unix_s = time.time()
                self.last_error = None
                if self.status == 'read_error':
                    self.status = 'ready'
                return self.latest.copy()
            except Exception as exc:
                self.last_error = str(exc)
                self.status = 'read_error'
                self._warn(f'[WARN] Tactile read failed: {exc}')
                return self.latest.copy()

    def total_norm(self):
        with self.lock:
            self.read()
            return float(self.latest_norm)

    def snapshot(self, *, refresh=True):
        with self.lock:
            if refresh:
                self.read()
            return {'values': np.asarray(self.latest, dtype=np.float32).copy(), 'num_mags': int(self.num_mags), 'total_norm': float(self.latest_norm), 'release_ref_norm': self.release_reference_norm, 'release_delta_norm': self.release_delta_norm, 'status': str(self.release_status), 'error': self.last_error, 'timestamp_perf_s': float(self.last_sample_perf_s), 'timestamp_unix_s': float(self.last_sample_unix_s)}

    def clear_runtime_state(self, *, release_status='reset_cleared'):
        with self.lock:
            self.release_reference_norm = None
            self.release_delta_norm = None
            self.release_status = str(release_status)
            self.latest = np.zeros(self.num_mags * 3, dtype=np.float32)
            self.latest_norm = 0.0
            self.last_sample_perf_s = time.perf_counter()
            self.last_sample_unix_s = time.time()
            self.last_error = None

    def set_release_reference(self, reference_norm):
        with self.lock:
            self.release_reference_norm = None if reference_norm is None else float(reference_norm)
            self.release_delta_norm = None
            self.release_status = 'armed' if reference_norm is not None else 'off'

    def update_release_delta(self, current_norm):
        with self.lock:
            if self.release_reference_norm is None:
                self.release_delta_norm = None
                return None
            self.release_delta_norm = abs(float(current_norm) - float(self.release_reference_norm))
            return self.release_delta_norm

    def close(self):
        with self.lock:
            stream = self.stream
            self.stream = None
        if stream is None:
            self.status = 'closed'
            return
        try:
            stream.pause_streaming()
        except Exception:
            pass
        try:
            stream.join()
        except Exception:
            pass
        self.status = 'closed'
        if self.debug and self.enabled:
            print('[Tactile] AnySkin stream stopped.')

def clamp_value(v, low, high):
    return max(low, min(high, v))

def clamp_pose_mm(x_mm, y_mm, z_mm, args):
    x_mm = clamp_value(x_mm, args.workspace_x[0], args.workspace_x[1])
    y_mm = clamp_value(y_mm, args.workspace_y[0], args.workspace_y[1])
    z_mm = clamp_value(z_mm, args.workspace_z[0], args.workspace_z[1])
    return (x_mm, y_mm, z_mm)

def meters_to_mm(values):
    return np.asarray(values, dtype=np.float32) * 1000.0

def mm_to_m_tuple(values):
    # Keep taught workspace endpoints exact through the mm -> m boundary.
    vec = np.asarray(values, dtype=np.float64).reshape(3) / 1000.0
    return tuple((float(v) for v in vec))

def compute_dynamic_eef_target(reference_xyz_mm, eef_xyz_mm):
    reference_xyz_mm = np.asarray(reference_xyz_mm, dtype=np.float32).reshape(3)
    offset = np.array([EEF_X_OFFSET_MM, EEF_Y_OFFSET_MM, EEF_Z_OFFSET_MM], dtype=np.float32)
    distance = float(np.linalg.norm(reference_xyz_mm[:2] - np.asarray(eef_xyz_mm)[:2]))
    far_distance = max(float(np.linalg.norm(offset)), 40.001)
    fraction = float(np.clip((distance - 40.0) / (far_distance - 40.0), 0.0, 1.0))
    target = reference_xyz_mm + fraction * offset
    return (target, float(fraction * offset[0]), distance)

def get_close_range_step_mm(ref_err_xyz, max_step_mm, max_step_z_mm):
    """Use smaller servo steps near the object for gentler final alignment."""
    dist_xy = float(np.linalg.norm(np.asarray(ref_err_xyz, dtype=np.float32)[:2]))
    if dist_xy < 95.0:
        return (5, 2.2, dist_xy)
    return (float(max_step_mm), float(max_step_z_mm), dist_xy)

def compute_close_range_ref_step_xyz(ref_err_xyz, max_step_xy, max_step_z, *, follow_z):
    """Compute one bounded servo step, staging close XY motion on a dominant axis."""
    ref_err_xyz = np.asarray(ref_err_xyz, dtype=np.float32).reshape(3)
    ref_step_xyz = np.zeros(3, dtype=np.float32)
    dist_xy = float(np.linalg.norm(ref_err_xyz[:2]))
    dominant_axis = None
    if dist_xy < 65.0 and abs(float(ref_err_xyz[0])) > 15.0 and (abs(float(ref_err_xyz[1])) > 15.0):
        dominant_axis = 0 if abs(float(ref_err_xyz[0])) >= abs(float(ref_err_xyz[1])) else 1
        ref_step_xyz[dominant_axis] = np.clip(ref_err_xyz[dominant_axis], -max_step_xy, max_step_xy)
    else:
        ref_step_xyz[0:2] = np.clip(ref_err_xyz[0:2], -max_step_xy, max_step_xy)
    if follow_z:
        ref_step_xyz[2] = np.clip(ref_err_xyz[2], -max_step_z, max_step_z)
    return (ref_step_xyz, dominant_axis)

def get_home_joints_rad():
    """Return the configured HOME joint target in radians."""
    return tuple((float(np.deg2rad(value)) for value in HOME_JOINTS_DEG))

def make_robot_command(command_type, *, target_position_base=None, fixed_orientation_base=None, gripper_action=None, source_mode='manual'):
    """Create the shared robot command object expected by XArmController.step()."""
    return RobotCommandState(command_type=command_type, target_position_base=target_position_base, fixed_orientation_base=fixed_orientation_base, gripper_action=gripper_action, source_mode=source_mode, stop_requested=command_type == ROBOT_CMD_STOP, timestamp=time.time(), valid=True)

def send_robot_command(controller, command_type, *, target_position_base=None, fixed_orientation_base=None, gripper_action=None, source_mode='manual'):
    """Send one robot command through the xArm controller with a fresh timestamp."""
    command = make_robot_command(command_type, target_position_base=target_position_base, fixed_orientation_base=fixed_orientation_base, gripper_action=gripper_action, source_mode=source_mode)
    return controller.step(command, now_timestamp=command.timestamp)

def init_xarm(args):
    """Connect xArm, validate robot state, and open the gripper for task startup."""
    print(f'[INFO] Connecting to xArm7 xArm using config {args.config} ...')
    controller = XArmController.from_config(args.config)
    if args.robot_ip:
        controller.robot_ip = args.robot_ip
    controller.fixed_orientation_format = 'rpy'
    try:
        controller.connect()
        state = controller.read_robot_state(now_timestamp=time.time())
        if not state.is_connected or not state.valid:
            raise RuntimeError(f'Failed to connect xArm controller. last_error={state.last_error}')
        if state.actual_tcp_pose_base is None:
            raise RuntimeError('xArm connected but actual_tcp_pose_base is unavailable.')
        pose = state.actual_tcp_pose_base
        print(f'[INFO] Current pose base: x={pose[0]:.4f}, y={pose[1]:.4f}, z={pose[2]:.4f}, rx={pose[3]:.4f}, ry={pose[4]:.4f}, rz={pose[5]:.4f}')
        if args.move_to_base:
            print('[INFO] --move-to-base is ignored; HOME is defined by HOME_JOINTS_DEG.')
        print('[INFO] Preparing Robotiq and opening gripper at startup...')
        if not controller.open_gripper_blocking():
            raise RuntimeError('Robotiq startup open did not complete')
        print('[INFO] Robotiq startup open complete; proceeding to HOME')
        if args.open_gripper:
            print('[INFO] --open-gripper specified; startup open was already applied.')
        return controller
    except BaseException:
        disconnect_xarm(controller)
        raise

def safe_stop_xarm(controller):
    if controller is None:
        return
    try:
        send_robot_command(controller, ROBOT_CMD_STOP, source_mode='manual_stop')
    except Exception:
        pass

def disconnect_xarm(controller):
    if controller is None:
        return
    try:
        controller.close()
    except Exception:
        pass

class FollowSharedState:
    """Thread-safe task state shared by the perception loop and robot follow thread."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.target_predictor = TargetPredictor(process_noise_mm_s2=args.prediction_process_noise_mm_s2, measurement_noise_mm=args.prediction_measurement_noise_mm, max_velocity_xy_mm_s=args.prediction_max_xy_speed_mm_s, reinit_jump_mm=args.prediction_reinit_jump_mm)
        self.follow_enabled = args.enable_follow
        self.latest_target_xyz_mm = None
        self.latest_grasp_xyz_mm = None
        self.latest_measurement_source = 'none'
        self.latest_target_t = 0.0
        self.last_measured_target_xyz_mm = None
        self.last_measured_target_t = 0.0
        self.valid_detection_streak = 0
        self.prediction_armed = False
        self.predicted_target_xyz_mm = None
        self.predicted_target_t = 0.0
        self.prediction_age_s = None
        self.control_target_xyz_mm = None
        self.target_source = 'none'
        self.reference_object_xy_mm = None
        self.reference_object_xyz_mm = None
        self.reference_locked = False
        self.motion_triggered = False
        self.reference_streak = 0
        self.initial_object_label = None
        self.fixed_orientation_base = None
        self.initial_pose_base = None
        self.stop_event = threading.Event()
        self.follow_pause_requested = False
        self.follow_idle_event = threading.Event()
        self.follow_idle_event.set()
        self.object_stopped = False
        self.stop_pose_buffer = deque(maxlen=15)
        self.latest_object_xyz_mm = None
        self.pregrasp_started = False
        self.grasp_closed = False
        self.grasp_offset_xyz_mm = None
        self.recent_grasp_z_mm_buffer = deque(maxlen=PLACE_Z_GRASP_BUFFER_FRAMES)
        self.recent_template_bottom_z_mm_buffer = deque(maxlen=PLACE_Z_GRASP_BUFFER_FRAMES)
        self.frozen_place_z_mm = None
        self.frozen_place_z_raw_mm = None
        self.frozen_place_z_grasp_median_mm = None
        self.frozen_place_z_template_bottom_median_mm = None
        self.frozen_place_z_used_fallback = True
        self.task_state = 'FOLLOW'
        self.task_epoch = 0

    def set_fixed_pose_from_robot(self, controller):
        """Capture z and orientation from the robot for follow-mode constraints."""
        state = controller.read_robot_state(now_timestamp=time.time())
        pose = state.actual_tcp_pose_base
        if pose is None:
            raise RuntimeError('Failed to get current robot pose for fixed pose.')
        with self.lock:
            self.fixed_orientation_base = tuple((float(v) for v in pose[3:6]))
            self.initial_pose_base = tuple((float(v) for v in pose))
        print(f'[INFO] Fixed pose set from current robot pose: rpy=({pose[3]:.4f}, {pose[4]:.4f}, {pose[5]:.4f})')

    def _reset_prediction_locked(self, *, reset_arm=False):
        self.target_predictor.reset()
        self.predicted_target_xyz_mm = None
        self.predicted_target_t = 0.0
        self.prediction_age_s = None
        self.control_target_xyz_mm = None
        self.target_source = 'none'
        if reset_arm:
            self.prediction_armed = False

    def set_task_state(self, task_state, *, reset_prediction=False, reset_arm=False):
        with self.lock:
            self.task_state = str(task_state)
            if reset_prediction:
                self._reset_prediction_locked(reset_arm=reset_arm)

    def set_follow_enabled(self, follow_enabled, *, reset_prediction=True, reset_arm=True):
        with self.lock:
            self.follow_enabled = bool(follow_enabled)
            if reset_prediction:
                self._reset_prediction_locked(reset_arm=reset_arm)
            print(f'[INFO] follow_enabled = {self.follow_enabled}')

    def toggle_follow(self):
        with self.lock:
            self.follow_enabled = not self.follow_enabled
            self._reset_prediction_locked(reset_arm=True)
            print(f'[INFO] follow_enabled = {self.follow_enabled}')

    def stop_follow(self):
        with self.lock:
            self.follow_enabled = False
            self._reset_prediction_locked(reset_arm=True)
            print('[INFO] Robot follow stopped.')

    def clear_target(self, *, reset_prediction=False, reset_arm=False):
        with self.lock:
            self.latest_target_xyz_mm = None
            self.latest_grasp_xyz_mm = None
            self.latest_measurement_source = 'none'
            self.valid_detection_streak = 0
            self.reference_streak = 0
            self.control_target_xyz_mm = None
            self.target_source = 'none'
            if reset_prediction:
                self._reset_prediction_locked(reset_arm=reset_arm)

    def update_place_z_samples(self, *, grasp_point_base=None, object_point_base=None, fitted_points_base=None):
        """Collect recent grasp and fitted-template z samples for later place height."""
        if getattr(self.args, 'place_object_xyz_mm', None) is not None:
            return
        grasp_z_mm = None
        source_point = grasp_point_base if grasp_point_base is not None else object_point_base
        if source_point is not None:
            source_point_arr = np.asarray(source_point, dtype=np.float32).reshape(3)
            if np.all(np.isfinite(source_point_arr)):
                grasp_z_mm = float(source_point_arr[2] * 1000.0)
        template_bottom_z_mm = None
        if fitted_points_base is not None:
            fitted_points = np.asarray(fitted_points_base, dtype=np.float32).reshape((-1, 3))
            if len(fitted_points) > 0:
                valid_z = fitted_points[np.isfinite(fitted_points[:, 2]), 2]
                if len(valid_z) > 0:
                    template_bottom_z_mm = float(np.min(valid_z) * 1000.0)
        with self.lock:
            if grasp_z_mm is not None:
                self.recent_grasp_z_mm_buffer.append(grasp_z_mm)
            if template_bottom_z_mm is not None:
                self.recent_template_bottom_z_mm_buffer.append(template_bottom_z_mm)

    def finalize_place_z_from_recent_samples(self):
        """Freeze the place height estimate before the gripper closes."""
        with self.lock:
            self.frozen_place_z_mm = None
            self.frozen_place_z_raw_mm = None
            self.frozen_place_z_grasp_median_mm = None
            self.frozen_place_z_template_bottom_median_mm = None
            self.frozen_place_z_used_fallback = True
            fixed_xyz = getattr(self.args, 'place_object_xyz_mm', None)
            if fixed_xyz is not None:
                self.frozen_place_z_mm = float(fixed_xyz[2])
                self.frozen_place_z_raw_mm = self.frozen_place_z_mm
                self.frozen_place_z_used_fallback = False
                return {'valid': True, 'reason': 'fixed_xyz', 'place_z_mm': self.frozen_place_z_mm}
            grasp_samples = np.asarray(self.recent_grasp_z_mm_buffer, dtype=np.float32)
            template_bottom_samples = np.asarray(self.recent_template_bottom_z_mm_buffer, dtype=np.float32)
            if len(grasp_samples) < PLACE_Z_MIN_VALID_SAMPLES or len(template_bottom_samples) < PLACE_Z_MIN_VALID_SAMPLES:
                return {'valid': False, 'reason': 'insufficient_samples', 'grasp_sample_count': int(len(grasp_samples)), 'template_bottom_sample_count': int(len(template_bottom_samples)), 'grasp_z_median_mm': None, 'template_bottom_z_median_mm': None, 'raw_place_z_mm': None, 'place_z_mm': None}
            grasp_z_median_mm = float(np.median(grasp_samples))
            template_bottom_z_median_mm = float(np.median(template_bottom_samples))
            release_parameter_mm = max(0.0, float(getattr(self.args, 'release_parameter_mm', RELEASE_PARAMETER_MM)))
            raw_place_z_mm = float(grasp_z_median_mm - template_bottom_z_median_mm + release_parameter_mm)
            place_z_mm = max(raw_place_z_mm, HOME_PLACE_MIN_Z_MM)
            self.frozen_place_z_mm = place_z_mm
            self.frozen_place_z_raw_mm = raw_place_z_mm
            self.frozen_place_z_grasp_median_mm = grasp_z_median_mm
            self.frozen_place_z_template_bottom_median_mm = template_bottom_z_median_mm
            self.frozen_place_z_used_fallback = False
            return {'valid': True, 'reason': 'ok', 'grasp_sample_count': int(len(grasp_samples)), 'template_bottom_sample_count': int(len(template_bottom_samples)), 'grasp_z_median_mm': grasp_z_median_mm, 'template_bottom_z_median_mm': template_bottom_z_median_mm, 'release_parameter_mm': release_parameter_mm, 'raw_place_z_mm': raw_place_z_mm, 'place_z_mm': place_z_mm}

    def request_follow_pause(self):
        with self.lock:
            self.follow_pause_requested = True

    def clear_follow_pause(self):
        with self.lock:
            self.follow_pause_requested = False

    def reset_for_restart(self, *, follow_enabled=None):
        """Clear task-local state so the next handover starts from a clean epoch."""
        with self.lock:
            if follow_enabled is None:
                follow_enabled = self.args.enable_follow
            self.follow_enabled = bool(follow_enabled)
            self.latest_target_xyz_mm = None
            self.latest_grasp_xyz_mm = None
            self.latest_measurement_source = 'none'
            self.latest_target_t = 0.0
            self.last_measured_target_xyz_mm = None
            self.last_measured_target_t = 0.0
            self.valid_detection_streak = 0
            self.prediction_armed = False
            self.reference_object_xy_mm = None
            self.reference_object_xyz_mm = None
            self.reference_locked = False
            self.motion_triggered = False
            self.reference_streak = 0
            self.initial_object_label = None
            self.fixed_orientation_base = None
            self.initial_pose_base = None
            self.follow_pause_requested = False
            self.object_stopped = False
            self.stop_pose_buffer.clear()
            self.latest_object_xyz_mm = None
            self.pregrasp_started = False
            self.grasp_closed = False
            self.grasp_offset_xyz_mm = None
            self.recent_grasp_z_mm_buffer.clear()
            self.recent_template_bottom_z_mm_buffer.clear()
            self.frozen_place_z_mm = None
            self.frozen_place_z_raw_mm = None
            self.frozen_place_z_grasp_median_mm = None
            self.frozen_place_z_template_bottom_median_mm = None
            self.frozen_place_z_used_fallback = True
            self.task_state = 'FOLLOW'
            self.task_epoch += 1
            self._reset_prediction_locked(reset_arm=True)
        self.follow_idle_event.set()
        print(f'[INFO] Follow/task state reset. follow_enabled={self.follow_enabled}')

    def set_follow_thread_idle(self, is_idle):
        if is_idle:
            self.follow_idle_event.set()
        else:
            self.follow_idle_event.clear()

    def wait_for_follow_idle(self, timeout_s):
        return self.follow_idle_event.wait(timeout=max(float(timeout_s), 0.0))

    def update_target(self, grasp_xyz_m, object_xyz_m=None, pixel_xy=None, eef_xyz_mm=None, measurement_source='measured', object_label=None):
        """Update measured/fallback target state and arm follow once motion is detected."""
        if object_xyz_m is None:
            self.clear_target(reset_prediction=False, reset_arm=False)
            return
        measurement_source = str(measurement_source or 'measured').strip().lower()
        if measurement_source not in {'measured', 'hand_fallback'}:
            measurement_source = 'measured'
        normalized_object_label = None
        if object_label is not None:
            normalized_object_label = str(object_label).strip().lower() or None
        object_xyz_mm = np.asarray(object_xyz_m, dtype=np.float32)[:3] * 1000.0
        object_xy_mm = object_xyz_mm[:2].copy()
        grasp_xyz_mm = None
        tracking_reference_xyz_mm = object_xyz_mm.copy()
        target_xyz_mm = None
        if grasp_xyz_m is not None:
            grasp_xyz_mm = np.asarray(grasp_xyz_m, dtype=np.float32)[:3] * 1000.0
            tracking_reference_xyz_mm = grasp_xyz_mm.copy()
        if eef_xyz_mm is None:
            target_xyz_mm = tracking_reference_xyz_mm.copy()
            target_xyz_mm[0] += EEF_X_OFFSET_MM
            target_xyz_mm[1] += EEF_Y_OFFSET_MM
        else:
            target_xyz_mm, dynamic_offset_x, dist_xy = compute_dynamic_eef_target(tracking_reference_xyz_mm, eef_xyz_mm)
            if self.args.verbose_robot:
                target_kind = 'grasp' if grasp_xyz_mm is not None else 'object'
                print(f'[ROBOT] tracking={target_kind}, dynamic_offset_x={dynamic_offset_x:.1f} mm, eef_target_dist_xy={dist_xy:.1f} mm')
        self.update_stop_state(object_xyz_mm)
        with self.lock:
            current_perf = time.perf_counter()
            self.latest_object_xyz_mm = object_xyz_mm.copy()
            self.latest_grasp_xyz_mm = None if grasp_xyz_mm is None else grasp_xyz_mm.copy()
            self.latest_measurement_source = measurement_source
            if not self.reference_locked:
                self.reference_streak += 1
                if self.reference_streak >= REFERENCE_LOCK_COUNT:
                    self.reference_object_xy_mm = object_xy_mm.copy()
                    self.reference_object_xyz_mm = object_xyz_mm.copy()
                    self.reference_locked = True
                    self.motion_triggered = False
                    print(f'[INFO] Reference object position locked: {self.reference_object_xyz_mm}')
                self.latest_target_xyz_mm = None
                self.valid_detection_streak = 0
                self._reset_prediction_locked(reset_arm=True)
                return
            if measurement_source == 'measured' and (not self.motion_triggered) and (normalized_object_label is not None) and (self.initial_object_label != normalized_object_label):
                self.initial_object_label = normalized_object_label
                print(f'[INFO] Cached initial segmentation label for grasp threshold: label={self.initial_object_label}')
            if not self.motion_triggered:
                move_dist_mm = float(np.linalg.norm(object_xy_mm - self.reference_object_xy_mm))
                move_dist_z_mm = 0.0
                if self.reference_object_xyz_mm is not None:
                    move_dist_z_mm = float(abs(object_xyz_mm[2] - self.reference_object_xyz_mm[2]))
                if move_dist_mm >= MOTION_TRIGGER_MM or move_dist_z_mm >= MOTION_TRIGGER_Z_MM:
                    self.motion_triggered = True
                    print(f'[INFO] Object motion detected: xy={move_dist_mm:.2f} mm, z={move_dist_z_mm:.2f} mm -> follow start')
            self.latest_target_xyz_mm = target_xyz_mm.copy()
            self.valid_detection_streak += 1
            self.latest_target_t = current_perf
            if measurement_source == 'measured':
                self.last_measured_target_xyz_mm = target_xyz_mm.copy()
                self.last_measured_target_t = current_perf
            if self.valid_detection_streak >= int(self.args.min_valid_count):
                self.prediction_armed = True
            if measurement_source == 'measured' and self.args.enable_target_prediction and (self.task_state == 'FOLLOW'):
                self.target_predictor.update(target_xyz_mm, current_perf)

    def _refresh_tracking_targets_locked(self, now_perf):
        """Select the freshest control target, falling back to short-horizon prediction."""
        measurement_age_s = None
        if self.last_measured_target_t > 0.0:
            measurement_age_s = max(float(now_perf) - float(self.last_measured_target_t), 0.0)
        self.predicted_target_xyz_mm = None
        self.predicted_target_t = 0.0
        self.prediction_age_s = None
        self.control_target_xyz_mm = None
        self.target_source = 'none'
        if self.task_state != 'FOLLOW':
            return measurement_age_s
        if self.args.enable_target_prediction and self.target_predictor.has_state():
            predicted = self.target_predictor.predict(now_perf)
            if predicted is not None and predicted.valid:
                self.predicted_target_xyz_mm = predicted.xyz_mm.copy()
                self.predicted_target_t = float(now_perf)
                self.prediction_age_s = float(predicted.prediction_age_s)
        raw_target = None if self.latest_target_xyz_mm is None else self.latest_target_xyz_mm.copy()
        raw_is_fresh = raw_target is not None and self.latest_target_t > 0.0 and (float(now_perf) - float(self.latest_target_t) <= float(self.args.target_timeout_s))
        if raw_is_fresh:
            self.control_target_xyz_mm = raw_target
            self.target_source = str(self.latest_measurement_source or 'measured')
            return measurement_age_s
        predicted_is_valid = self.args.enable_target_prediction and self.prediction_armed and (self.predicted_target_xyz_mm is not None) and (self.prediction_age_s is not None) and (float(self.prediction_age_s) <= float(self.args.prediction_max_horizon_s))
        if predicted_is_valid:
            self.control_target_xyz_mm = self.predicted_target_xyz_mm.copy()
            self.target_source = 'predicted'
        return measurement_age_s

    def get_snapshot(self, now_perf=None):
        """Return a consistent copy of follow/task state for readers outside the lock."""
        with self.lock:
            now_perf = time.perf_counter() if now_perf is None else float(now_perf)
            measurement_age_s = self._refresh_tracking_targets_locked(now_perf)
            return {'follow_enabled': self.follow_enabled, 'latest_target_xyz_mm': None if self.latest_target_xyz_mm is None else self.latest_target_xyz_mm.copy(), 'latest_grasp_xyz_mm': None if self.latest_grasp_xyz_mm is None else self.latest_grasp_xyz_mm.copy(), 'measurement_source': self.latest_measurement_source, 'latest_target_t': self.latest_target_t, 'last_measured_target_xyz_mm': None if self.last_measured_target_xyz_mm is None else self.last_measured_target_xyz_mm.copy(), 'last_measured_target_t': self.last_measured_target_t, 'valid_detection_streak': self.valid_detection_streak, 'prediction_armed': self.prediction_armed, 'predicted_target_xyz_mm': None if self.predicted_target_xyz_mm is None else self.predicted_target_xyz_mm.copy(), 'predicted_target_t': self.predicted_target_t, 'prediction_age_s': self.prediction_age_s, 'control_target_xyz_mm': None if self.control_target_xyz_mm is None else self.control_target_xyz_mm.copy(), 'target_source': self.target_source, 'measurement_age_s': measurement_age_s, 'motion_triggered': self.motion_triggered, 'follow_pause_requested': self.follow_pause_requested, 'fixed_orientation_base': None if self.fixed_orientation_base is None else tuple(self.fixed_orientation_base), 'latest_object_xyz_mm': None if self.latest_object_xyz_mm is None else self.latest_object_xyz_mm.copy(), 'initial_object_label': self.initial_object_label, 'object_stopped': self.object_stopped, 'pregrasp_started': self.pregrasp_started, 'initial_pose_base': None if self.initial_pose_base is None else tuple(self.initial_pose_base), 'task_state': self.task_state, 'task_epoch': self.task_epoch}

    def update_stop_state(self, object_xyz_mm):
        """Detect when the object has settled after motion starts."""
        if not self.motion_triggered:
            self.object_stopped = False
            self.stop_pose_buffer.clear()
            return
        self.stop_pose_buffer.append(object_xyz_mm.copy())
        if len(self.stop_pose_buffer) < self.stop_pose_buffer.maxlen:
            self.object_stopped = False
            return
        buf = np.stack(self.stop_pose_buffer, axis=0)
        xyz_range = buf.max(axis=0) - buf.min(axis=0)
        stable_xy = xyz_range[0] < 30.0 and xyz_range[1] < 30.0
        stable_z = xyz_range[2] < 50.0
        if stable_xy and stable_z:
            if not self.object_stopped:
                print(f'[INFO] Object STOPPED detected. xyz_range={xyz_range}')
            self.object_stopped = True
        else:
            self.object_stopped = False

    def is_pregrasp_pose_reached(self, eef_pose_base, x_tol_mm=None, y_tol_mm=None, z_tol_mm=None):
        """Check whether a robot pose is close enough to close the gripper."""
        snapshot = self.get_snapshot()
        if not snapshot['follow_enabled'] or snapshot['follow_pause_requested'] or not snapshot['motion_triggered']:
            return False
        if snapshot['valid_detection_streak'] < self.args.min_valid_count or snapshot['target_source']=='predicted':
            return False
        target_xyz = snapshot['control_target_xyz_mm']
        if target_xyz is None or eef_pose_base is None:
            return False
        configured=self.args.pregrasp_tolerance_mm
        x_tol_mm=configured[0] if x_tol_mm is None else x_tol_mm
        y_tol_mm=configured[1] if y_tol_mm is None else y_tol_mm
        z_tol_mm=configured[2] if z_tol_mm is None else z_tol_mm
        eef_x, eef_y, eef_z = meters_to_mm(eef_pose_base[:3])
        obj_x, obj_y, obj_z = target_xyz[:3]
        dx = float(obj_x - eef_x)
        dy = float(obj_y - eef_y)
        dz = float(obj_z - eef_z)
        x_ok = abs(dx) <= x_tol_mm
        y_ok = abs(dy) <= y_tol_mm
        z_ok = abs(dz) <= z_tol_mm
        return x_ok and y_ok and z_ok

    def should_start_pregrasp(self, controller, x_tol_mm=None, y_tol_mm=None, z_tol_mm=None):
        """Compatibility wrapper for tests and legacy callers."""
        state = controller.read_robot_state(now_timestamp=time.time())
        return self.is_pregrasp_pose_reached(state.actual_tcp_pose_base, x_tol_mm=x_tol_mm, y_tol_mm=y_tol_mm, z_tol_mm=z_tol_mm)

class RobotWorker:
    """Single thread that owns all xArm and gripper commands."""

    def __init__(self, args, shared_state, metadata_recorder=None, tactile_manager=None):
        self.args = args
        self.shared_state = shared_state
        self.metadata_recorder = metadata_recorder
        self.tactile_manager = tactile_manager
        self.controller = None
        self._request_queue = queue.Queue()
        self._status = RobotStatus()
        self._status_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._thread = None
        self._request_seq = 0
        self._request_seq_lock = threading.Lock()
        self._last_sent_pose_mm = None
        self._ref_target_xyz_mm = None
        self._was_follow_active = False
        self._next_follow_tick_t = 0.0
        self._last_status_read_t = 0.0

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name='robot-worker', daemon=True)
        self._thread.start()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def submit(self, request):
        if not isinstance(request, RobotRequest):
            request = RobotRequest(str(request))
        payload = {} if request.payload is None else dict(request.payload)
        with self._request_seq_lock:
            self._request_seq += 1
            request_id = self._request_seq
        queued_request = RobotRequest(type=str(request.type), payload=payload, created_at=float(request.created_at), request_id=request_id)
        if queued_request.type in ROBOT_URGENT_REQUESTS:
            self._cancel_event.set()
        if queued_request.type in {ROBOT_REQ_INIT_ROBOT,ROBOT_REQ_RESET_HOME,ROBOT_REQ_SHUTDOWN,ROBOT_REQ_EMERGENCY_STOP}:
            self._set_status(home_ready=False)
        self._request_queue.put(queued_request)
        return request_id

    def get_status(self):
        with self._status_lock:
            return replace(self._status)

    def _set_status(self, **updates):
        with self._status_lock:
            for key, value in updates.items():
                setattr(self._status, key, value)

    def _bump_status_counter(self, field_name):
        with self._status_lock:
            setattr(self._status, field_name, int(getattr(self._status, field_name)) + 1)
            return int(getattr(self._status, field_name))

    def _current_state(self):
        with self._status_lock:
            return self._status.state

    def _set_active_request(self, request):
        self._set_status(active_request=request.type, active_request_id=int(request.request_id))

    def _clear_active_request(self):
        self._set_status(active_request=None, active_request_id=0)

    def _reset_follow_tracking(self):
        self._last_sent_pose_mm = None
        self._ref_target_xyz_mm = None
        self._was_follow_active = False
        self.shared_state.set_follow_thread_idle(True)

    def _publish_robot_state(self, state):
        if state is None:
            return
        updates = {}
        if hasattr(state, 'actual_tcp_pose_base'):
            pose = getattr(state, 'actual_tcp_pose_base', None)
            updates['last_robot_pose'] = None if pose is None else tuple((float(v) for v in pose))
        if hasattr(state, 'is_connected'):
            updates['is_connected'] = bool(getattr(state, 'is_connected', False))
        using_mock = getattr(state, 'using_mock', getattr(self.controller, 'using_mock', None))
        if using_mock is not None:
            updates['using_mock'] = bool(using_mock)
        if hasattr(state, 'last_error'):
            updates['last_error'] = getattr(state, 'last_error', None)
        if updates:
            self._set_status(**updates)
        self._last_status_read_t = time.time()

    def _read_robot_state(self):
        if self.controller is None:
            self._set_status(is_connected=False, using_mock=False)
            return None
        try:
            state = self.controller.read_robot_state(now_timestamp=time.time())
        except Exception as exc:
            self._set_status(last_error=str(exc), is_connected=False)
            return None
        self._publish_robot_state(state)
        return state

    def _send_robot_command(self, command_type, **kwargs):
        if self.controller is None:
            return None
        self._set_status(last_command_type=command_type)
        return send_robot_command(self.controller, command_type, **kwargs)

    def _safe_stop(self, source_mode='worker_stop'):
        if self.controller is None:
            return
        try:
            self._send_robot_command(ROBOT_CMD_STOP, source_mode=source_mode)
        except Exception as exc:
            self._set_status(last_error=str(exc))

    def _disconnect(self):
        if self.controller is None:
            return
        disconnect_xarm(self.controller)
        self.controller = None
        self._set_status(is_connected=False, using_mock=False, last_robot_pose=None)

    def _publish_task_ready(self):
        with self._status_lock:
            self._status.home_ready=True
            self._status.task_ready_epoch+=1

    def _run(self):
        while not self._shutdown_event.is_set():
            try:
                request = self._request_queue.get(timeout=0.01)
                self._handle_request(request)
                if request.type == ROBOT_REQ_SHUTDOWN:
                    break
                continue
            except queue.Empty:
                pass
            try:
                if self._current_state() == ROBOT_STATE_FOLLOWING:
                    self._follow_once()
                elif self.controller is not None and time.time() - self._last_status_read_t >= 0.1:
                    self._read_robot_state()
            except Exception as exc:
                print(f'[ERROR] Robot worker failed: {exc}',flush=True)
                self._safe_stop('worker_failed')
                self.shared_state.stop_follow()
                self._set_status(state=ROBOT_STATE_ERROR, last_error=str(exc))
        self._shutdown_event.set()

    def _handle_request(self, request):
        self._set_active_request(request)
        try:
            if request.type == ROBOT_REQ_INIT_ROBOT:
                self._handle_init_robot()
            elif request.type == ROBOT_REQ_START_FOLLOW:
                self._handle_start_follow()
            elif request.type == ROBOT_REQ_STOP_FOLLOW:
                self._handle_stop_follow()
            elif request.type == ROBOT_REQ_START_GRASP_PLACE:
                self._handle_start_grasp_place(request.payload)
            elif request.type == ROBOT_REQ_RESET_HOME:
                self._handle_reset_home()
            elif request.type == ROBOT_REQ_SAVE_AND_STOP:
                self._handle_save_and_stop()
            elif request.type == ROBOT_REQ_EMERGENCY_STOP:
                self._handle_emergency_stop()
            elif request.type == ROBOT_REQ_SHUTDOWN:
                self._handle_shutdown()
            else:
                self._reject_request(request, f'unknown request type {request.type}')
        except Exception as exc:
            print(f'[ERROR] Robot request {request.type} failed: {exc}',flush=True)
            self._safe_stop('request_failed')
            self.shared_state.clear_target(reset_prediction=True, reset_arm=True)
            self.shared_state.stop_follow()
            self._set_status(state=ROBOT_STATE_ERROR, last_error=str(exc))
        finally:
            self._clear_active_request()

    def _reject_request(self, request, reason):
        message = f'Rejected robot request {request.type}: {reason}'
        print(f'[WARN] {message}')
        self._set_status(last_error=message)

    def _handle_init_robot(self):
        if self.controller is not None:
            self._set_status(last_error=None)
            return
        self._cancel_event.clear()
        self._set_status(state=ROBOT_STATE_INITIALIZING, last_error=None, grasp_ok=None)
        try:
            self.controller = init_xarm(self.args)
            self._read_robot_state()
            home_ok = move_robot_to_home_pose(self.controller, self.args, cancel_event=self._cancel_event, on_state_read=self._publish_robot_state)
            if home_ok is False:
                self.shared_state.stop_follow()
                self._disconnect()
                self._set_status(state=ROBOT_STATE_IDLE, last_error='robot initialization cancelled')
                return
            self.shared_state.set_fixed_pose_from_robot(self.controller)
            self._read_robot_state()
            self._publish_task_ready()
            self._set_status(state=ROBOT_STATE_IDLE, last_error=None)
        except Exception as exc:
            self.shared_state.stop_follow()
            self._safe_stop('initialization_failed')
            self._disconnect()
            self._set_status(state=ROBOT_STATE_ERROR, last_error=str(exc))
            print(f'[WARN] Robot initialization failed: {exc}')

    def _handle_start_follow(self):
        if self.controller is None:
            detail = self.get_status().last_error
            reason = 'robot is not initialized' if detail is None else f'robot is not initialized: {detail}'
            self._reject_request(RobotRequest(ROBOT_REQ_START_FOLLOW), reason)
            return
        state = self._current_state()
        if state in {ROBOT_STATE_INITIALIZING, ROBOT_STATE_ERROR, ROBOT_STATE_GRASPING, ROBOT_STATE_RETURNING, ROBOT_STATE_PLACING, ROBOT_STATE_RESETTING, ROBOT_STATE_STOPPING}:
            self._reject_request(RobotRequest(ROBOT_REQ_START_FOLLOW), f'state is {state}')
            return
        self._cancel_event.clear()
        self.shared_state.clear_follow_pause()
        self.shared_state.set_follow_enabled(True, reset_prediction=True, reset_arm=True)
        self.shared_state.set_task_state('FOLLOW', reset_prediction=False, reset_arm=False)
        self._reset_follow_tracking()
        self._next_follow_tick_t = 0.0
        self._set_status(state=ROBOT_STATE_FOLLOWING, last_error=None, grasp_ok=None)

    def _handle_stop_follow(self):
        self.shared_state.request_follow_pause()
        self.shared_state.stop_follow()
        if self._was_follow_active:
            self._safe_stop(source_mode='stop_follow')
        self._reset_follow_tracking()
        if self._current_state() == ROBOT_STATE_FOLLOWING:
            self._set_status(state=ROBOT_STATE_IDLE)

    def _handle_start_grasp_place(self, payload):
        if self.controller is None:
            self._reject_request(RobotRequest(ROBOT_REQ_START_GRASP_PLACE), 'robot is not initialized')
            return
        if self._current_state() != ROBOT_STATE_FOLLOWING:
            self._reject_request(RobotRequest(ROBOT_REQ_START_GRASP_PLACE), f'state is {self._current_state()}')
            return
        robot_state = self._read_robot_state()
        robot_pose = None if robot_state is None else getattr(robot_state, 'actual_tcp_pose_base', None)
        if not self.shared_state.is_pregrasp_pose_reached(robot_pose):
            self._set_status(last_error='grasp request ignored because final pose check failed')
            return
        with self.shared_state.lock:
            frozen_place_z_mm = self.shared_state.frozen_place_z_mm
        if frozen_place_z_mm is None or not np.isfinite(float(frozen_place_z_mm)):
            self._set_status(last_error='grasp request ignored because place Z is not ready')
            return
        context = payload.get('context')
        if context is None:
            context = RobotActionContext()
        elif isinstance(context, dict):
            context = RobotActionContext(**context)
        self._set_status(home_ready=False)
        self.shared_state.request_follow_pause()
        self.shared_state.stop_follow()
        self._safe_stop(source_mode='pregrasp_handoff')
        self._reset_follow_tracking()
        tactile_enabled = self.tactile_manager is not None and bool(getattr(self.tactile_manager, 'enabled', False))
        grasp_detection_mode = getattr(self.args, 'grasp_detection_mode', GRASP_DETECTION_HYBRID)
        if grasp_detection_mode == GRASP_DETECTION_ROBOTIQ_OBJECT_ONLY:
            print('[INFO] Tactile-sensing ablation active; gripper geometry, xArm force/current, position threshold, and stall detection will not determine grasp success.')
        elif tactile_enabled:
            reset_gripper_position_threshold_to_config_default(self.controller)
        else:
            configure_gripper_position_threshold_from_geometry(self.controller, context.fitted_points_base, context.grasp_point_base, object_label=context.object_label, template_axes_base=context.template_axes_base)
        self._set_status(state=ROBOT_STATE_GRASPING, grasp_ok=None, last_error=None)
        grasp_ok = execute_gripper_close(self.controller, timeout_s=self.args.gripper_close_timeout_s, verbose=True, metadata_recorder=self.metadata_recorder, tactile_manager=self.tactile_manager, tactile_contact_threshold=self.args.tactile_contact_norm_threshold, tactile_extra_grasp_pos=self.args.tactile_extra_grasp_pos, detection_mode=grasp_detection_mode, cancel_event=self._cancel_event, on_state_read=self._publish_robot_state)
        self._set_status(grasp_ok=bool(grasp_ok))
        print(f'[INFO] grasp_ok = {grasp_ok}')
        if self._cancel_event.is_set():
            self._safe_stop(source_mode='grasp_cancelled')
            self._set_status(state=ROBOT_STATE_IDLE)
            return
        if not grasp_ok:
            link_error = getattr(self.controller, 'gripper_communication_error', None)
            detail = str(link_error) if isinstance(link_error, Exception) else 'gripper close did not verify grasp'
            self._set_status(state=ROBOT_STATE_ERROR, last_error=detail)
            return
        save_grasp_offset(self.controller, self.shared_state, on_state_read=self._publish_robot_state)
        if self._cancel_event.is_set():
            self._safe_stop(source_mode='post_grasp_cancelled')
            self._set_status(state=ROBOT_STATE_IDLE)
            return
        self._set_status(state=ROBOT_STATE_RETURNING)
        self._set_status(state=ROBOT_STATE_PLACING)
        place_ok = execute_return_and_place(self.controller, self.shared_state, self.args, metadata_recorder=self.metadata_recorder, tactile_manager=self.tactile_manager, cancel_event=self._cancel_event, on_state_read=self._publish_robot_state)
        if self._cancel_event.is_set():
            self._safe_stop(source_mode='place_cancelled')
            self._set_status(state=ROBOT_STATE_IDLE)
            return
        if not place_ok:
            self._set_status(state=ROBOT_STATE_ERROR, last_error='return/place failed')
            return
        self._bump_status_counter('task_done_epoch')
        self._set_status(state=ROBOT_STATE_DONE, last_error=None)

    def _handle_reset_home(self):
        self._cancel_event.clear()
        self._set_status(state=ROBOT_STATE_RESETTING, last_error=None, grasp_ok=None)
        try:
            self.shared_state.request_follow_pause()
            self.shared_state.stop_follow()
            self._safe_stop(source_mode='reset_home')
            self._reset_follow_tracking()
            if self.controller is not None:
                execute_gripper_open(self.controller, dwell_s=self.args.gripper_release_dwell_s, cancel_event=self._cancel_event)
                home_ok = move_robot_to_home_pose(self.controller, self.args, cancel_event=self._cancel_event, on_state_read=self._publish_robot_state)
                if home_ok is False:
                    self._set_status(state=ROBOT_STATE_IDLE, last_error='reset cancelled')
                    return
                reset_tactile_state_for_system_reset(self.tactile_manager)
                self.shared_state.reset_for_restart(follow_enabled=self.args.enable_follow)
                self.shared_state.set_fixed_pose_from_robot(self.controller)
                self.shared_state.clear_follow_pause()
                self._read_robot_state()
                self._publish_task_ready()
                self._set_status(state=ROBOT_STATE_FOLLOWING if self.args.enable_follow else ROBOT_STATE_IDLE)
            else:
                reset_tactile_state_for_system_reset(self.tactile_manager)
                self.shared_state.reset_for_restart(follow_enabled=False)
                self._set_status(state=ROBOT_STATE_IDLE)
            self._bump_status_counter('reset_done_epoch')
        except Exception as exc:
            self._set_status(state=ROBOT_STATE_ERROR, last_error=str(exc))
            print(f'[WARN] Robot reset failed: {exc}')

    def _handle_save_and_stop(self):
        self._set_status(state=ROBOT_STATE_STOPPING)
        self.shared_state.request_follow_pause()
        self.shared_state.stop_follow()
        self._safe_stop(source_mode='save_and_stop')
        self._reset_follow_tracking()
        self._cancel_event.clear()
        self._set_status(state=ROBOT_STATE_IDLE)

    def _handle_emergency_stop(self):
        self._set_status(state=ROBOT_STATE_STOPPING)
        self.shared_state.request_follow_pause()
        self.shared_state.stop_follow()
        self._safe_stop(source_mode='emergency_stop')
        self._reset_follow_tracking()

    def _handle_shutdown(self):
        self._set_status(state=ROBOT_STATE_STOPPING)
        self.shared_state.request_follow_pause()
        self.shared_state.stop_follow()
        self._safe_stop(source_mode='shutdown')
        self._reset_follow_tracking()
        self._disconnect()
        self._shutdown_event.set()

    def _follow_once(self):
        now = time.time()
        interval = 1.0 / max(float(self.args.control_hz), 1e-06)
        if now < self._next_follow_tick_t:
            return
        self._next_follow_tick_t = now + interval
        robot_state = self._read_robot_state()
        if robot_state is None or not robot_state.valid:
            self._safe_stop('invalid_feedback')
            self.shared_state.stop_follow()
            self._set_status(state=ROBOT_STATE_ERROR, last_error='Invalid xArm feedback')
            return
        snap = self.shared_state.get_snapshot()
        control_target_xyz_mm = snap['control_target_xyz_mm']
        target_source = snap['target_source']
        active = True
        if snap['follow_pause_requested']:
            active = False
        elif not snap['follow_enabled']:
            active = False
        elif control_target_xyz_mm is None:
            active = False
        elif target_source != 'predicted' and snap['valid_detection_streak'] < self.args.min_valid_count and (not snap['prediction_armed']):
            active = False
        elif target_source == 'predicted' and (not snap['prediction_armed']):
            active = False
        elif not snap['motion_triggered']:
            active = False
        elif target_source == 'predicted' and (snap['prediction_age_s'] is None or float(snap['prediction_age_s']) > float(self.args.prediction_max_horizon_s)):
            active = False
        elif snap['fixed_orientation_base'] is None:
            active = False
        if not active:
            if self._was_follow_active:
                self._safe_stop(source_mode='follow_inactive')
                self._ref_target_xyz_mm = None
                self._last_sent_pose_mm = None
            self._was_follow_active = False
            self.shared_state.set_follow_thread_idle(True)
            return
        self.shared_state.set_follow_thread_idle(False)
        fixed_orientation_base = snap['fixed_orientation_base']
        if self._ref_target_xyz_mm is None:
            pose = None if robot_state is None else getattr(robot_state, 'actual_tcp_pose_base', None)
            if pose is None:
                return
            self._ref_target_xyz_mm = meters_to_mm(pose[:3]).astype(np.float32)
            print(f'[INFO] ref_target initialized from current EEF xyz: {self._ref_target_xyz_mm}')
        max_step_mm = MAX_XY_SPEED_MM_S / max(float(self.args.control_hz), 1e-06)
        max_step_z_mm = MAX_Z_SPEED_MM_S / max(float(self.args.control_hz), 1e-06)
        ref_err_xyz = control_target_xyz_mm - self._ref_target_xyz_mm
        max_step_xy, max_step_z, dist_xy = get_close_range_step_mm(ref_err_xyz, max_step_mm, max_step_z_mm)
        ref_step_xyz, dominant_axis = compute_close_range_ref_step_xyz(ref_err_xyz, max_step_xy, max_step_z, follow_z=bool(self.args.follow_z))
        self._ref_target_xyz_mm = self._ref_target_xyz_mm + ref_step_xyz
        cmd_z = float(self._ref_target_xyz_mm[2])
        cmd_x, cmd_y, cmd_z = clamp_pose_mm(float(self._ref_target_xyz_mm[0]), float(self._ref_target_xyz_mm[1]), cmd_z, self.args)
        pose_mm = np.array([cmd_x, cmd_y, cmd_z], dtype=np.float32)
        if self._last_sent_pose_mm is not None:
            pos_delta = np.linalg.norm(pose_mm - self._last_sent_pose_mm)
            if pos_delta < 0.2:
                self.controller.heartbeat()
                return
        target_position_base = mm_to_m_tuple(pose_mm)
        try:
            self._send_robot_command(ROBOT_CMD_SERVO_TO_POSITION, target_position_base=target_position_base, fixed_orientation_base=fixed_orientation_base, gripper_action=GRIPPER_HOLD, source_mode='follow_servo')
            self._was_follow_active = True
            self._last_sent_pose_mm = pose_mm
        except Exception as exc:
            self._set_status(last_error=str(exc))
            self._safe_stop('servo_failure')
            self.shared_state.stop_follow()
            self._set_status(state=ROBOT_STATE_ERROR)
            print(f'[WARN] servo command failed: {exc}')

def compute_tcp_speed_norms(state):
    """Return linear/angular TCP speed norms from a RobotState-like object."""
    speed = getattr(state, 'actual_tcp_speed', None)
    if speed is None or len(speed) < 3:
        return (None, None)
    try:
        linear_values = np.asarray(speed[:3], dtype=np.float64)
        linear_norm = float(np.linalg.norm(linear_values))
        angular_norm = None
        if len(speed) >= 6:
            angular_values = np.asarray(speed[3:6], dtype=np.float64)
            angular_norm = float(np.linalg.norm(angular_values))
        return (linear_norm, angular_norm)
    except Exception:
        return (None, None)

def read_xarm_is_steady(controller):
    """Return xArm isSteady() when available; otherwise None."""
    xarm_control = controller
    is_steady = getattr(xarm_control, 'isSteady', None)
    if not callable(is_steady):
        return None
    try:
        return bool(is_steady())
    except Exception:
        return None

def wait_until_robot_stopped(controller, *, timeout_s, speed_threshold_mps=DEFAULT_POST_BACKOFF_STOP_SPEED_THRESHOLD_MPS, poll_dt=DEFAULT_POST_BACKOFF_STOP_POLL_DT_S, cancel_event=None, on_state_read=None, source_mode='robot_stop_wait'):
    """Poll robot state until TCP speed or xArm steady status indicates a stop."""
    del source_mode
    if cancel_event is not None and cancel_event.is_set():
        safe_stop_xarm(controller)
        return False
    if not hasattr(controller, 'read_robot_state'):
        return False
    deadline = time.time() + max(0.0, float(timeout_s))
    threshold = max(0.0, float(speed_threshold_mps))
    poll_dt = max(0.0, float(poll_dt))
    while True:
        if cancel_event is not None and cancel_event.is_set():
            safe_stop_xarm(controller)
            return False
        try:
            state = controller.read_robot_state(now_timestamp=time.time())
            if on_state_read is not None:
                on_state_read(state)
        except Exception:
            state = None
        if state is not None:
            linear_speed, _angular_speed = compute_tcp_speed_norms(state)
            if linear_speed is not None and linear_speed <= threshold:
                return True
        is_steady = read_xarm_is_steady(controller)
        if is_steady is True:
            return True
        if time.time() >= deadline:
            return False
        time.sleep(poll_dt)

def wait_until_target_reached(controller, target_position_base, *, timeout_s, tolerance_m, poll_dt=0.05, cancel_event=None, on_state_read=None):
    """Poll the robot pose until a blocking move reaches its target or times out."""
    deadline = time.time() + timeout_s
    target = np.asarray(target_position_base, dtype=np.float32).reshape(3)
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            safe_stop_xarm(controller)
            return False
        state = controller.read_robot_state(now_timestamp=time.time())
        if on_state_read is not None:
            on_state_read(state)
        if not state.valid:
            safe_stop_xarm(controller)
            return False
        pose = state.actual_tcp_pose_base
        if pose is not None:
            current = np.asarray(pose[:3], dtype=np.float32).reshape(3)
            error = float(np.linalg.norm(current - target))
            if error <= tolerance_m:
                return True
        time.sleep(poll_dt)
    return False

def wait_until_joint_target_reached(controller, target_joints_rad, *, timeout_s, tolerance_rad, poll_dt=0.05, cancel_event=None, on_state_read=None):
    """Poll the robot joints until the target is reached or the move times out."""
    deadline = time.time() + timeout_s
    target = np.asarray(target_joints_rad, dtype=np.float64).reshape(7)
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            if hasattr(controller, 'stop_joint_motion'):
                controller.stop_joint_motion()
            else:
                safe_stop_xarm(controller)
            return False
        state = controller.read_robot_state(now_timestamp=time.time())
        if on_state_read is not None:
            on_state_read(state)
        if not state.valid:
            safe_stop_xarm(controller)
            return False
        joints = getattr(state, 'joint_positions', None)
        if joints is not None:
            current = np.asarray(joints, dtype=np.float64).reshape(7)
            if np.all(np.isfinite(current)):
                max_error = float(np.max(np.abs(current - target)))
                if max_error <= tolerance_rad:
                    if read_xarm_is_steady(controller) is not False:
                        return True
        time.sleep(poll_dt)
    return False

def move_robot_and_wait(controller, target_position_base, fixed_orientation_base, *, timeout_s, tolerance_m, source_mode, cancel_event=None, on_state_read=None):
    """Issue one blocking position move and wait for completion."""
    if cancel_event is not None and cancel_event.is_set():
        safe_stop_xarm(controller)
        return False
    send_robot_command(controller, ROBOT_CMD_MOVE_TO_POSITION, target_position_base=target_position_base, fixed_orientation_base=fixed_orientation_base, gripper_action=GRIPPER_HOLD, source_mode=source_mode)
    ok = wait_until_target_reached(controller, target_position_base, timeout_s=timeout_s, tolerance_m=tolerance_m, cancel_event=cancel_event, on_state_read=on_state_read)
    if not ok:
        safe_stop_xarm(controller)
    return ok

def stop_follow_for_handoff(shared_state, timeout_s):
    """Pause the follow thread before the main thread takes over for grasp/place."""
    shared_state.request_follow_pause()
    released = shared_state.wait_for_follow_idle(timeout_s)
    if not released:
        print(f'[WARN] Follow thread did not release robot control within {timeout_s:.2f}s')
        return False
    print('[INFO] Follow thread released robot control for pregrasp handoff')
    return True

def move_robot_to_home_pose(controller, args, cancel_event=None, on_state_read=None):
    """Move the robot to the configured HOME joint target before or between tasks."""
    if cancel_event is not None and cancel_event.is_set():
        safe_stop_xarm(controller)
        return False
    home_joints_rad = get_home_joints_rad()
    home_text = ', '.join((f'{value:.1f}' for value in HOME_JOINTS_DEG))
    print(f'[INFO] Moving robot to HOME joints: deg=[{home_text}], speed={HOME_JOINT_SPEED_RAD_S:.2f} rad/s, accel={HOME_JOINT_ACCELERATION_RAD_S2:.2f} rad/s^2')
    if not hasattr(controller, 'move_to_joint_positions'):
        raise RuntimeError('xArm controller does not support joint HOME moves.')
    started = bool(controller.move_to_joint_positions(home_joints_rad, speed_rad_s=HOME_JOINT_SPEED_RAD_S, acceleration_rad_s2=HOME_JOINT_ACCELERATION_RAD_S2, async_move=True))
    if not started:
        raise RuntimeError('Failed to start HOME joint move.')
    ok = wait_until_joint_target_reached(controller, home_joints_rad, timeout_s=args.move_timeout_s, tolerance_rad=float(np.deg2rad(HOME_JOINT_TOLERANCE_DEG)), cancel_event=cancel_event, on_state_read=on_state_read)
    if not ok:
        if hasattr(controller, 'stop_joint_motion'):
            controller.stop_joint_motion()
        else:
            safe_stop_xarm(controller)
        raise RuntimeError('Failed to reach HOME joints during startup.')
    print('[INFO] HOME joints reached')
    return True

def resolve_default_gripper_position_threshold(gripper_cfg):
    """Resolve the fallback gripper close completion threshold."""
    gripper_cfg = dict(gripper_cfg or {})
    return int(gripper_cfg.get('position_complete_threshold', DEFAULT_GRIPPER_POSITION_COMPLETE_THRESHOLD))

def resolve_gripper_position_stall_detection_config(controller):
    """Resolve gripper position stall fallback settings from controller config."""
    config = getattr(controller, 'config', {}) or {}
    gripper_cfg = dict(config.get('robot', {}).get('gripper', {}) or {})
    stall_cfg = dict(gripper_cfg.get('position_stall_detection', {}) or {})
    return {'enabled': bool(stall_cfg.get('enabled', DEFAULT_GRIPPER_POSITION_STALL_ENABLED)), 'stable_reads_required': max(1, int(stall_cfg.get('stable_reads_required', DEFAULT_GRIPPER_POSITION_STALL_STABLE_READS))), 'tolerance': max(0, int(stall_cfg.get('tolerance', DEFAULT_GRIPPER_POSITION_STALL_TOLERANCE))), 'min_elapsed_s': max(0.0, float(stall_cfg.get('min_elapsed_s', DEFAULT_GRIPPER_POSITION_STALL_MIN_ELAPSED_S)))}

def _gripper_geometry_cfg(gripper_cfg):
    gripper_cfg = dict(gripper_cfg or {})
    geometry_cfg = gripper_cfg.get('position_threshold_geometry', {})
    return dict(geometry_cfg) if isinstance(geometry_cfg, dict) else {}

def _estimate_xy_max_diameter_m(points_xy, max_points=128):
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape((-1, 2))
    if len(points_xy) == 0:
        return None
    if len(points_xy) > int(max_points):
        sample_indices = np.linspace(0, len(points_xy) - 1, int(max_points), dtype=np.int32)
        points_xy = points_xy[sample_indices]
    if len(points_xy) == 1:
        return 0.0
    deltas = points_xy[:, None, :] - points_xy[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    return float(np.max(distances))

def _estimate_pca_section_widths_m(points_xy):
    points_xy = np.asarray(points_xy, dtype=np.float64).reshape((-1, 2))
    if len(points_xy) < 2:
        return None
    centered = points_xy - np.mean(points_xy, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    if not np.all(np.isfinite(covariance)):
        return None
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    basis = eigvecs[:, order]
    projected = centered @ basis
    extents = np.ptp(projected, axis=0)
    if len(extents) < 2 or not np.all(np.isfinite(extents)):
        return None
    long_width = float(np.max(extents))
    short_width = float(np.min(extents))
    return (short_width, long_width)

def _normalize_gripper_label(label):
    return '' if label is None else str(label).strip().lower()

def _local_width_axis_for_label(geometry_cfg, label):
    axis_by_label = geometry_cfg.get('local_width_axis_by_label', {})
    if not isinstance(axis_by_label, dict):
        return None
    normalized_label = _normalize_gripper_label(label)
    for configured_label, configured_axis in axis_by_label.items():
        if _normalize_gripper_label(configured_label) == normalized_label:
            axis = str(configured_axis).strip().lower()
            return axis if axis in {'x', 'y', 'z'} else None
    return None

def _estimate_local_axis_width_m(slice_points, template_axes_base, axis_name):
    points = np.asarray(slice_points, dtype=np.float64).reshape((-1, 3))
    try:
        axes = np.asarray(template_axes_base, dtype=np.float64).reshape((3, 3))
    except Exception:
        return None
    axis_index = {'x': 0, 'y': 1, 'z': 2}.get(str(axis_name).strip().lower())
    if axis_index is None or len(points) == 0 or (not np.all(np.isfinite(axes))):
        return None
    axis = axes[axis_index]
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 1e-09:
        return None
    axis = axis / norm
    projected = points @ axis.reshape(3, 1)
    projected = projected.reshape(-1)
    if len(projected) == 0 or not np.all(np.isfinite(projected)):
        return None
    return float(np.max(projected) - np.min(projected))

def _select_template_slice_points(fitted_points_base, grasp_point_base, geometry_cfg):
    if fitted_points_base is None:
        return (None, 'missing_geometry')
    points = np.asarray(fitted_points_base, dtype=np.float64).reshape((-1, 3))
    if len(points) == 0 or grasp_point_base is None:
        return (None, 'missing_geometry')
    grasp = np.asarray(grasp_point_base, dtype=np.float64).reshape(-1)
    if len(grasp) < 3 or not np.isfinite(grasp[:3]).all():
        return (None, 'missing_grasp')
    z_values = points[:, 2]
    finite_mask = np.isfinite(z_values) & np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    points = points[finite_mask]
    if len(points) == 0:
        return (None, 'missing_geometry')
    grasp_z = float(grasp[2])
    slice_band_m = max(float(geometry_cfg.get('slice_band_m', 0.005)), 0.0)
    min_slice_points = max(int(geometry_cfg.get('min_slice_points', 8)), 1)
    half_band_m = 0.5 * slice_band_m
    if half_band_m > 0.0:
        z_delta = np.abs(points[:, 2] - grasp_z)
        slice_points = points[z_delta <= half_band_m]
    else:
        slice_points = np.empty((0, 3), dtype=np.float64)
    if len(slice_points) >= min_slice_points:
        return (slice_points, 'band')
    fallback_count = max(int(geometry_cfg.get('fallback_nearest_points', 64)), min_slice_points)
    nearest_count = min(fallback_count, len(points))
    if nearest_count <= 0:
        return (None, 'missing_geometry')
    nearest_indices = np.argsort(np.abs(points[:, 2] - grasp_z))[:nearest_count]
    nearest_points = points[nearest_indices]
    if len(nearest_points) < min_slice_points:
        return (None, 'insufficient_slice_points')
    return (nearest_points, 'nearest_z')

def estimate_gripper_template_width_cm(fitted_points_base, grasp_point_base, gripper_cfg, *, object_label=None, template_axes_base=None):
    """Estimate template width/diameter at the grasp z position in centimeters."""
    geometry_cfg = _gripper_geometry_cfg(gripper_cfg)
    slice_points, slice_source = _select_template_slice_points(fitted_points_base, grasp_point_base, geometry_cfg)
    if slice_points is None:
        return (None, {'reason': slice_source, 'slice_source': slice_source, 'slice_point_count': 0})
    local_axis = _local_width_axis_for_label(geometry_cfg, object_label)
    if local_axis is not None:
        if template_axes_base is None:
            return (None, {'reason': 'missing_template_axes', 'slice_source': slice_source, 'slice_point_count': int(len(slice_points)), 'width_mode': f'local_axis_{local_axis}', 'label': _normalize_gripper_label(object_label) or None})
        width_m = _estimate_local_axis_width_m(slice_points, template_axes_base, local_axis)
        if width_m is None or not np.isfinite(width_m):
            return (None, {'reason': 'invalid_local_axis_width', 'slice_source': slice_source, 'slice_point_count': int(len(slice_points)), 'width_mode': f'local_axis_{local_axis}', 'label': _normalize_gripper_label(object_label) or None})
        width_cm = float(width_m * 100.0)
        return (width_cm, {'reason': 'ok', 'slice_source': slice_source, 'slice_point_count': int(len(slice_points)), 'width_mode': f'local_axis_{local_axis}', 'local_axis': local_axis, 'label': _normalize_gripper_label(object_label) or None, 'width_cm': width_cm})
    points_xy = slice_points[:, :2]
    diameter_m = _estimate_xy_max_diameter_m(points_xy)
    pca_widths = _estimate_pca_section_widths_m(points_xy)
    width_mode = 'diameter'
    width_m = diameter_m
    anisotropic_ratio = max(float(geometry_cfg.get('anisotropic_ratio', 1.25)), 1.0)
    if pca_widths is not None:
        short_width_m, long_width_m = pca_widths
        if short_width_m > 1e-09 and long_width_m / short_width_m >= anisotropic_ratio:
            width_m = short_width_m
            width_mode = 'pca_short_axis'
    if width_m is None or not np.isfinite(width_m):
        return (None, {'reason': 'invalid_width', 'slice_source': slice_source, 'slice_point_count': int(len(slice_points))})
    width_cm = float(width_m * 100.0)
    return (width_cm, {'reason': 'ok', 'slice_source': slice_source, 'slice_point_count': int(len(slice_points)), 'width_mode': width_mode, 'width_cm': width_cm})

def resolve_gripper_position_threshold_from_geometry(gripper_cfg, fitted_points_base, grasp_point_base, *, object_label=None, template_axes_base=None):
    """Resolve gripper close completion threshold from fitted template geometry."""
    gripper_cfg = dict(gripper_cfg or {})
    default_threshold = resolve_default_gripper_position_threshold(gripper_cfg)
    geometry_cfg = _gripper_geometry_cfg(gripper_cfg)
    if not bool(geometry_cfg.get('enabled', True)):
        return (default_threshold, {'reason': 'disabled', 'threshold_float': float(default_threshold), 'threshold': int(default_threshold), 'fallback_threshold': int(default_threshold)})
    width_cm, debug = estimate_gripper_template_width_cm(fitted_points_base, grasp_point_base, gripper_cfg, object_label=object_label, template_axes_base=template_axes_base)
    if width_cm is None:
        debug.update({'threshold_float': float(default_threshold), 'threshold': int(default_threshold), 'fallback_threshold': int(default_threshold)})
        return (default_threshold, debug)
    formula_opening_cm = float(geometry_cfg.get('formula_opening_cm', 9.0))
    if not np.isfinite(formula_opening_cm) or formula_opening_cm <= 1e-09:
        formula_opening_cm = 9.0
    alpha_cm = float(geometry_cfg.get('alpha_cm', 0.1))
    threshold_float = (formula_opening_cm - (float(width_cm) + alpha_cm)) * 255.0 / formula_opening_cm
    clamp_min = float(geometry_cfg.get('clamp_min', 0.0))
    clamp_max = float(geometry_cfg.get('clamp_max', 255.0))
    if clamp_min > clamp_max:
        clamp_min, clamp_max = (clamp_max, clamp_min)
    threshold_clamped = float(np.clip(threshold_float, clamp_min, clamp_max))
    threshold = int(np.floor(threshold_clamped + 0.5 + 1e-09))
    debug.update({'alpha_cm': alpha_cm, 'formula_opening_cm': formula_opening_cm, 'threshold_float': threshold_float, 'threshold_clamped_float': threshold_clamped, 'threshold': threshold, 'fallback_threshold': int(default_threshold)})
    return (threshold, debug)

def configure_gripper_position_threshold_from_geometry(controller, fitted_points_base, grasp_point_base, *, object_label=None, template_axes_base=None):
    """Select a gripper completion threshold from the fitted template grasp section."""
    if controller is None:
        return None
    controller_config = getattr(controller, 'config', {}) or {}
    gripper_cfg = controller_config.get('robot', {}).get('gripper', {})
    position_threshold, debug = resolve_gripper_position_threshold_from_geometry(gripper_cfg, fitted_points_base, grasp_point_base, object_label=object_label, template_axes_base=template_axes_base)
    controller.gripper_position_complete_threshold = int(position_threshold)
    reason = debug.get('reason', 'unknown')
    width_cm = debug.get('width_cm')
    width_text = 'n/a' if width_cm is None else f'{float(width_cm):.3f}cm'
    threshold_float = float(debug.get('threshold_float', position_threshold))
    threshold_clamped = float(debug.get('threshold_clamped_float', position_threshold))
    print(f"[INFO] Gripper position threshold configured from geometry: threshold_float={threshold_float:.3f}, threshold_clamped={threshold_clamped:.3f}, threshold={int(position_threshold)}, width={width_text}, mode={debug.get('width_mode', 'fallback')}, label={debug.get('label', _normalize_gripper_label(object_label) or 'unknown')}, slice_source={debug.get('slice_source', 'n/a')}, slice_points={int(debug.get('slice_point_count', 0))}, reason={reason}")
    return int(position_threshold)

def reset_gripper_position_threshold_to_config_default(controller):
    """Use the static gripper threshold fallback without vision/template geometry."""
    if controller is None:
        return None
    controller_config = getattr(controller, 'config', {}) or {}
    gripper_cfg = controller_config.get('robot', {}).get('gripper', {})
    position_threshold = int(gripper_cfg.get('position_complete_threshold', DEFAULT_GRIPPER_POSITION_COMPLETE_THRESHOLD))
    controller.gripper_position_complete_threshold = int(position_threshold)
    print(f'[INFO] Tactile mode enabled; skipping vision/template gripper threshold geometry. Using static fallback position threshold={int(position_threshold)}.')
    return int(position_threshold)

def start_task_video_recording(video_recorder, task_ready_timestamp):
    if video_recorder is None:
        return
    ok, message = video_recorder.start_recording_for_task(task_start_timestamp_iso=task_ready_timestamp)
    level = '[INFO]' if ok else '[WARN]'
    print(f'{level} Video recorder: {message}')

def open_video_recorder_ui(video_recorder):
    if video_recorder is None:
        return
    opened, url = video_recorder.open_browser()
    if opened:
        print(f'[INFO] Video recorder UI opened: {url}')
    else:
        print(f'[INFO] Open the video recorder UI to finalize save: {url}')

def discard_video_recording_for_reset(video_recorder):
    if video_recorder is None:
        return
    ok, message = video_recorder.discard_pending_recording()
    if ok:
        print(f'[INFO] Video recorder reset discard: {message}')
    elif message != 'nothing to discard':
        print(f'[WARN] Video recorder reset discard failed: {message}')
    else:
        print('[INFO] Video recorder reset discard: nothing to discard')
    try:
        status = video_recorder.get_status()
    except Exception as exc:
        print(f'[WARN] Video recorder status check failed after reset: {exc}')
        return
    has_live_frame = bool(status.get('has_live_frame'))
    last_frame_age_ms = status.get('last_frame_age_ms')
    if has_live_frame:
        age_text = '-' if last_frame_age_ms is None else f'{int(last_frame_age_ms)} ms'
        print(f'[INFO] Video recorder kept live stream active after reset (last_frame_age={age_text}).')
    else:
        print('[WARN] Video recorder has no live frame after reset; next task recording will wait for the next frame.')

def reset_tactile_state_for_system_reset(tactile_manager):
    if tactile_manager is None or not bool(getattr(tactile_manager, 'enabled', False)):
        return False
    if hasattr(tactile_manager, 'clear_runtime_state'):
        tactile_manager.clear_runtime_state(release_status='reset_cleared')
    else:
        num_mags = max(1, int(getattr(tactile_manager, 'num_mags', DEFAULT_TACTILE_NUM_MAGS)))
        tactile_manager.release_reference_norm = None
        tactile_manager.release_delta_norm = None
        tactile_manager.release_status = 'reset_cleared'
        tactile_manager.latest = np.zeros(num_mags * 3, dtype=np.float32)
        tactile_manager.latest_norm = 0.0
        tactile_manager.last_error = None
    baseline_reset = False
    if hasattr(tactile_manager, 'reset_baseline'):
        baseline_reset = bool(tactile_manager.reset_baseline())
    if hasattr(tactile_manager, 'clear_runtime_state'):
        tactile_manager.clear_runtime_state(release_status='reset_ready' if baseline_reset else 'reset_cleared')
    else:
        tactile_manager.release_reference_norm = None
        tactile_manager.release_delta_norm = None
        tactile_manager.release_status = 'reset_ready' if baseline_reset else 'reset_cleared'
    print(f'[Tactile] Reset state for system reset: baseline_reset={baseline_reset}, ref cleared, delta cleared.')
    return baseline_reset

def reset_perception_pipeline_for_system_reset(pipeline, preserve_tracking=False, preserve_fitting=False):
    """Reset perception-only state after the robot worker finishes a reset."""
    if pipeline is None:
        return
    shape_fitting_tracker = pipeline.get('shape_fitting_tracker')
    if not preserve_fitting and shape_fitting_tracker is not None and hasattr(shape_fitting_tracker, 'reset'):
        shape_fitting_tracker.reset()
        print('[INFO] Shape fitting tracker reset. ICP initialization will restart.')
    object_merger = pipeline.get('object_merger')
    if object_merger is not None and hasattr(object_merger, 'reset_initial_centroid'):
        object_merger.reset_initial_centroid()
    for object_worker_key in ('object_worker_cam0', 'object_worker_cam1'):
        object_worker = pipeline.get(object_worker_key)
        if not preserve_tracking and object_worker is not None and hasattr(object_worker, 'reset'):
            object_worker.reset()
    if preserve_tracking:
        print('[INFO] SAM2 tracking memory preserved.')
    else:
        print('[INFO] Object workers reset. Point prompts will be reacquired.')
    for hand_worker_key in ('hand_worker_cam0', 'hand_worker_cam1'):
        hand_worker = pipeline.get(hand_worker_key)
        if hand_worker is not None and hasattr(hand_worker, 'reset'):
            hand_worker.reset()
    print('[INFO] Hand workers reset. MediaPipe tracking and hand smoothing will restart.')
    hand_selector = pipeline.get('hand_selector')
    if hand_selector is not None and hasattr(hand_selector, 'reset'):
        hand_selector.reset()
        print('[INFO] Hand selector reset. Locked hand camera will be reacquired.')
    fusion = pipeline.get('fusion')
    if fusion is not None and hasattr(fusion, 'reset'):
        fusion.reset()
        print('[INFO] Perception fusion reset. Filtered hand/object centers cleared.')
    grasp_z_stabilizer = pipeline.get('grasp_z_stabilizer')
    if grasp_z_stabilizer is not None and hasattr(grasp_z_stabilizer, 'reset'):
        grasp_z_stabilizer.reset()
        print('[INFO] Grasp z stabilizer reset. Filtered grasp height cleared.')
    hand_relative_fallback = pipeline.get('hand_relative_fallback')
    if hand_relative_fallback is not None and hasattr(hand_relative_fallback, 'reset'):
        hand_relative_fallback.reset()

def begin_same_object_refit(pipeline, task_id):
    runtime=pipeline.get('sam3d_live_runtime')
    if runtime is None or not runtime.begin_refit(task_id):
        return False
    reset_perception_pipeline_for_system_reset(pipeline,preserve_tracking=True,preserve_fitting=True)
    print('[INFO] Existing SAM3D template retained; fresh SAM2 masks and ICP must be valid before follow resumes.')
    return True


def reset_system_to_start_state(controller, shared_state, args, metadata_recorder=None, pipeline=None, video_recorder=None, tactile_manager=None):
    """Reset perception, robot pose, metadata, and recorder state for a new task."""
    print('[INFO] Reset requested: returning to startup state')
    task_start_perf = None
    discard_video_recording_for_reset(video_recorder)
    reset_perception_pipeline_for_system_reset(pipeline)
    if controller is None:
        reset_tactile_state_for_system_reset(tactile_manager)
        shared_state.reset_for_restart(follow_enabled=False)
        return
    shared_state.request_follow_pause()
    shared_state.stop_follow()
    shared_state.wait_for_follow_idle(args.follow_handoff_timeout_s)
    safe_stop_xarm(controller)
    execute_gripper_open(controller, dwell_s=args.gripper_release_dwell_s)
    move_robot_to_home_pose(controller, args)
    reset_tactile_state_for_system_reset(tactile_manager)
    shared_state.reset_for_restart(follow_enabled=args.enable_follow)
    shared_state.set_fixed_pose_from_robot(controller)
    if metadata_recorder is not None:
        task_ready_timestamp = metadata_recorder.mark_task_ready(shared_state)
        task_start_perf = time.perf_counter()
        print(f'[INFO] Metadata task start timestamp={task_ready_timestamp}')
        start_task_video_recording(video_recorder, task_ready_timestamp)
    shared_state.clear_follow_pause()
    print('[INFO] Reset complete. System is back at startup state.')
    return task_start_perf

def stop_gripper_motion_safely(controller, reason):
    if not hasattr(controller, 'stop_gripper_motion'):
        return
    try:
        controller.stop_gripper_motion()
    except Exception as exc:
        print(f'[WARN] Failed to stop gripper after {reason}: {exc}')

def note_robot_first_contact(metadata_recorder):
    if metadata_recorder is None:
        return
    first_contact_timestamp = metadata_recorder.note_robot_first_contact()
    if first_contact_timestamp is not None:
        print(f'[INFO] Robot first contact timestamp={first_contact_timestamp}')

def print_tactile_frame_log(tactile_manager, *, stage, stage_time_s=None, **fields):
    if tactile_manager is None or not bool(getattr(tactile_manager, 'enabled', False)):
        return
    num_mags = max(1, int(getattr(tactile_manager, 'num_mags', DEFAULT_TACTILE_NUM_MAGS)))
    tactile_snapshot = None
    if hasattr(tactile_manager, 'snapshot'):
        tactile_snapshot = tactile_manager.snapshot(refresh=False)
    tactile_values = getattr(tactile_manager, 'latest', []) if tactile_snapshot is None else tactile_snapshot.get('values', [])
    tactile_data = np.asarray(tactile_values, dtype=np.float32).flatten()
    expected_values = num_mags * 3
    if tactile_data.size < expected_values:
        padded = np.zeros((expected_values,), dtype=np.float32)
        padded[:tactile_data.size] = tactile_data
        tactile_data = padded
    tactile_data = tactile_data[:expected_values].reshape(num_mags, 3)
    time_text = '-' if stage_time_s is None else f'{float(stage_time_s):.3f}'
    ref_norm = getattr(tactile_manager, 'release_reference_norm', None) if tactile_snapshot is None else tactile_snapshot.get('release_ref_norm')
    ref_text = '-' if ref_norm is None else f'{float(ref_norm):.3f}'
    delta_norm = getattr(tactile_manager, 'release_delta_norm', None) if tactile_snapshot is None else tactile_snapshot.get('release_delta_norm')
    delta_text = '-' if delta_norm is None else f'{float(delta_norm):.3f}'
    status = str(getattr(tactile_manager, 'release_status', 'off') if tactile_snapshot is None else tactile_snapshot.get('status', 'off'))
    norm = float(getattr(tactile_manager, 'latest_norm', 0.0) if tactile_snapshot is None else tactile_snapshot.get('total_norm', 0.0))
    field_text = ' '.join((f'{key}={value}' for key, value in fields.items() if value is not None))
    print(f'[TACTILE_FRAME] stage={stage} stage_time_s={time_text} norm={norm:.3f} ref={ref_text} delta={delta_text} status={status} {field_text}', flush=True)

def execute_gripper_close(controller, timeout_s=5.0, poll_dt=0.05, verbose=True, metadata_recorder=None, tactile_manager=None, tactile_contact_threshold=None, tactile_extra_grasp_pos=None, detection_mode=GRASP_DETECTION_HYBRID, cancel_event=None, on_state_read=None):
    """Close and verify contact using configured sensors and Robotiq feedback."""
    detection_mode = str(detection_mode).strip().lower()
    if detection_mode not in {GRASP_DETECTION_HYBRID, GRASP_DETECTION_ROBOTIQ_OBJECT_ONLY}:
        raise ValueError(f'Unsupported grasp detection mode: {detection_mode!r}')
    robotiq_object_only = detection_mode == GRASP_DETECTION_ROBOTIQ_OBJECT_ONLY
    if verbose:
        print('[INFO] GRIPPER CLOSE start')
    if cancel_event is not None and cancel_event.is_set():
        stop_gripper_motion_safely(controller, 'cancel before close')
        return False
    baseline_state = controller.read_robot_state(now_timestamp=time.time())
    if on_state_read is not None:
        on_state_read(baseline_state)
    if not baseline_state.valid:
        safe_stop_xarm(controller)
        return False
    baseline_force_norm = baseline_state.tcp_force_norm_n
    if verbose:
        print(f'[INFO] Pre-close baseline force_norm={baseline_force_norm}')
    try:
        if hasattr(controller, 'start_gripper_close'):
            if not controller.start_gripper_close():
                raise RuntimeError('gripper close command was rejected')
        else:
            send_robot_command(controller, ROBOT_CMD_HOLD, gripper_action=GRIPPER_CLOSE, source_mode='gripper_close')
    except Exception as exc:
        print(f'[ERROR] Gripper close command failed; no automatic resend: {exc}', flush=True)
        safe_stop_xarm(controller)
        return False
    deadline = time.time() + timeout_s
    loop_start = time.time()
    force_threshold = float(getattr(controller, 'min_tcp_force_norm_n', 8.0))
    stall_cfg = resolve_gripper_position_stall_detection_config(controller)
    last_position_value = None
    stable_position_reads = 0
    tactile_enabled = tactile_manager is not None and bool(getattr(tactile_manager, 'enabled', False))
    tactile_contact_threshold = DEFAULT_TACTILE_CONTACT_NORM_THRESHOLD if tactile_contact_threshold is None else float(tactile_contact_threshold)
    tactile_extra_grasp_pos = DEFAULT_TACTILE_EXTRA_GRASP_POS if tactile_extra_grasp_pos is None else max(0, int(tactile_extra_grasp_pos))
    tactile_contact_detected = False
    tactile_contact_position = None
    tactile_extra_target_position = None
    robotiq_command_armed = False
    robotiq_initial_position = None
    if tactile_enabled:
        tactile_manager.release_status = 'close_monitoring'
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            stop_gripper_motion_safely(controller, 'cancel')
            if tactile_enabled:
                tactile_manager.release_status = 'close_cancelled'
            return False
        state = controller.read_robot_state(now_timestamp=time.time())
        if on_state_read is not None:
            on_state_read(state)
        if not state.valid:
            stop_gripper_motion_safely(controller, 'invalid feedback')
            safe_stop_xarm(controller)
            return False
        force_norm = state.tcp_force_norm_n
        elapsed = time.time() - loop_start
        force_delta = None
        if baseline_force_norm is not None and force_norm is not None:
            force_delta = float(force_norm) - float(baseline_force_norm)
        force_triggered = False
        if elapsed >= GRIPPER_FORCE_STOP_MIN_ELAPSED_S and force_norm is not None:
            if baseline_force_norm is None:
                force_triggered = float(force_norm) >= force_threshold
            else:
                force_triggered = float(force_norm) >= force_threshold and force_delta is not None and (force_delta >= GRIPPER_FORCE_STOP_DELTA_N)
        gripper_close_state = None
        position_triggered = False
        position_stall_triggered = False
        position_value_int = None
        closed_position_int = 255
        if hasattr(controller, 'get_gripper_close_state'):
            try:
                gripper_close_state = controller.get_gripper_close_state()
                if gripper_close_state is not None:
                    position_value = gripper_close_state.get('position')
                    closed_position = gripper_close_state.get('closed_position')
                    if closed_position is not None:
                        closed_position_int = int(closed_position)
                    if position_value is not None:
                        position_value_int = int(position_value)
                        position_threshold = int(getattr(controller, 'gripper_position_complete_threshold', DEFAULT_GRIPPER_POSITION_COMPLETE_THRESHOLD))
                        position_triggered = position_value_int >= position_threshold
                        if last_position_value is None:
                            stable_position_reads = 1
                        elif abs(position_value_int - last_position_value) <= stall_cfg['tolerance']:
                            stable_position_reads += 1
                        else:
                            stable_position_reads = 1
                        last_position_value = position_value_int
                        position_stall_triggered = bool(stall_cfg['enabled']) and elapsed >= stall_cfg['min_elapsed_s'] and (stable_position_reads >= stall_cfg['stable_reads_required'])
            except Exception as exc:
                print(f'[ERROR] Gripper status read failed; aborting grasp without further gripper commands: {exc}', flush=True)
                if tactile_enabled:
                    tactile_manager.release_status = 'close_error'
                safe_stop_xarm(controller)
                return False
        elapsed = time.time() - loop_start
        object_status = None if gripper_close_state is None else str(gripper_close_state.get('object_status', '')).strip().upper()
        if object_status == 'MOVING':
            # A position threshold or repeated encoder value is not object
            # contact while Robotiq explicitly reports that it is still moving.
            position_triggered = False
            position_stall_triggered = False
        if object_status in {'AT_DEST', 'STOPPED_OUTER_OBJECT'}:
            if not robotiq_command_armed and elapsed < ROBOTIQ_OBJ_COMMAND_GRACE_S:
                time.sleep(poll_dt)
                continue
            position_triggered = False
            position_stall_triggered = False
            if robotiq_object_only:
                print(f'[WARN] Robotiq close finished without inner-object contact: status={object_status}, position={position_value_int}. Grasp failed; no additional gripper command will be sent.')
                return False
        if robotiq_object_only:
            if robotiq_initial_position is None and position_value_int is not None:
                robotiq_initial_position = int(position_value_int)
            position_delta = None
            if robotiq_initial_position is not None and position_value_int is not None:
                position_delta = int(position_value_int) - int(robotiq_initial_position)
            if object_status == 'MOVING' or (position_delta is not None and position_delta >= ROBOTIQ_OBJ_POSITION_START_DELTA):
                robotiq_command_armed = True
            if verbose:
                print(f"[GRIPPER][ROBOTIQ_OBJ_ONLY] object_status={object_status or 'unavailable'}, position={position_value_int}, position_delta={position_delta}, armed={robotiq_command_armed}, elapsed_s={elapsed:.3f}")
            if object_status == 'STOPPED_INNER_OBJECT':
                stop_gripper_motion_safely(controller, 'Robotiq OBJ inner-object detection')
                note_robot_first_contact(metadata_recorder)
                print('[INFO] Robotiq OBJ detected an object while closing; grasp verified.')
                return True
            time.sleep(poll_dt)
            continue
        tactile_norm = None
        tactile_triggered = False
        tactile_extra_done = False
        if tactile_enabled:
            tactile_norm = tactile_manager.total_norm()
            print_tactile_frame_log(tactile_manager, stage='gripper_close', stage_time_s=elapsed, gripper_pos=position_value_int)
            if not tactile_contact_detected and tactile_norm >= tactile_contact_threshold:
                tactile_contact_detected = True
                tactile_triggered = True
                tactile_contact_position = position_value_int
                if position_value_int is None:
                    stop_gripper_motion_safely(controller, 'tactile trigger without gripper position')
                    note_robot_first_contact(metadata_recorder)
                    tactile_manager.release_status = 'close_tactile_stop'
                    print(f'[INFO] Tactile contact detected during close (norm={tactile_norm:.3f} >= {tactile_contact_threshold:.3f}); gripper position unavailable, stopping close.')
                    return True
                tactile_extra_target_position = min(int(position_value_int) + int(tactile_extra_grasp_pos), int(closed_position_int))
                tactile_manager.release_status = 'close_extra'
                print(f'[INFO] Tactile contact detected during close: norm={tactile_norm:.3f}, contact_pos={position_value_int}, extra={tactile_extra_grasp_pos}, target_pos={tactile_extra_target_position}.')
            if tactile_contact_detected and tactile_extra_target_position is not None:
                tactile_extra_done = position_value_int is not None and int(position_value_int) >= int(tactile_extra_target_position)
                if tactile_extra_done:
                    stop_gripper_motion_safely(controller, 'tactile extra close')
                    note_robot_first_contact(metadata_recorder)
                    tactile_manager.release_status = 'close_done'
                    print(f'[INFO] Tactile extra close target reached. position={position_value_int}, target={tactile_extra_target_position}. Stopping gripper close and finishing grasp stage.')
                    return True
                if position_triggered:
                    position_triggered = False
        if verbose:
            tactile_text = ''
            if tactile_enabled:
                tactile_text = f', tactile_norm={tactile_norm}, tactile_triggered={tactile_triggered}, tactile_target={tactile_extra_target_position}'
            print(f'[GRIPPER] force_norm={state.tcp_force_norm_n}, force_delta={force_delta}, mean_joint_current={state.mean_joint_current_a}, verified={state.grasp_verified_force_current}, force_triggered={force_triggered}, position_triggered={position_triggered}, position_stall_triggered={position_stall_triggered}, stable_position_reads={stable_position_reads}, gripper_state={gripper_close_state}{tactile_text}')
        if force_triggered:
            stop_gripper_motion_safely(controller, 'force trigger')
            note_robot_first_contact(metadata_recorder)
            delta_str = 'n/a' if force_delta is None else f'{force_delta:.3f}'
            print(f'[INFO] Force rise detected during close (abs={float(force_norm):.3f} N, delta={delta_str} N). Stopping gripper close.')
            return True
        if object_status in {'AT_DEST', 'STOPPED_OUTER_OBJECT'}:
            # Preserve AnySkin/force evidence above, but a terminal encoder
            # position alone is not a successful grasp. It is already stopped.
            if tactile_enabled:
                tactile_manager.release_status = 'close_no_object'
            print(f'[WARN] Robotiq close finished without contact evidence: status={object_status}, position={position_value_int}. Grasp failed; no additional gripper command will be sent.')
            return False
        if position_triggered:
            stop_gripper_motion_safely(controller, 'position trigger')
            note_robot_first_contact(metadata_recorder)
            position_threshold = int(getattr(controller, 'gripper_position_complete_threshold', DEFAULT_GRIPPER_POSITION_COMPLETE_THRESHOLD))
            print(f'[INFO] Gripper position reached threshold >= {position_threshold}. Stopping gripper close and finishing grasp stage.')
            return True
        if position_stall_triggered:
            stop_gripper_motion_safely(controller, 'position stall trigger')
            note_robot_first_contact(metadata_recorder)
            if tactile_enabled and tactile_contact_detected:
                tactile_manager.release_status = 'close_stall_after_contact'
            print(f"[INFO] Gripper position stalled (position={position_value_int}, stable_reads={stable_position_reads}, tolerance={stall_cfg['tolerance']}). Stopping gripper close and finishing grasp stage.")
            return True
        time.sleep(poll_dt)
    stop_gripper_motion_safely(controller, 'close timeout')
    if tactile_enabled:
        tactile_manager.release_status = 'close_timeout'
    if robotiq_object_only:
        print('[WARN] Grasp could not be verified from the Robotiq OBJ signal before timeout.')
    else:
        print('[WARN] Grasp could not be verified from xArm force/current.')
    return False

def execute_gripper_open(controller, dwell_s=0.5, metadata_recorder=None, cancel_event=None):
    """Open the gripper and optionally record the last-contact timestamp."""
    if cancel_event is not None and cancel_event.is_set():
        stop_gripper_motion_safely(controller, 'cancel before open')
        return False
    print('[INFO] RELEASE opening gripper; waiting for confirmed open.',flush=True)
    try:
        send_robot_command(controller, ROBOT_CMD_HOLD, gripper_action=GRIPPER_OPEN, source_mode='gripper_open')
    except Exception as exc:
        safe_stop_xarm(controller)
        print(f'[ERROR] RELEASE open failed; backoff/HOME aborted: {exc}',flush=True)
        raise
    if metadata_recorder is not None:
        last_contact_timestamp = metadata_recorder.note_robot_last_contact()
        if last_contact_timestamp is not None:
            print(f'[INFO] Robot last contact timestamp={last_contact_timestamp}')
    print('[INFO] RELEASE gripper open confirmed.',flush=True)
    deadline = time.time() + max(dwell_s, 0.0)
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            stop_gripper_motion_safely(controller, 'cancel during open dwell')
            return False
        time.sleep(min(0.05, max(deadline - time.time(), 0.0)))
    return True

def capture_tactile_release_reference(tactile_manager, delay_s=0.0, cancel_event=None):
    """Capture the held-object tactile norm used to gate release timing."""
    if tactile_manager is None or not bool(getattr(tactile_manager, 'enabled', False)):
        return None
    if cancel_event is not None and cancel_event.is_set():
        return None
    delay_s = max(0.0, float(delay_s))
    if delay_s > 0.0:
        tactile_manager.release_status = 'ref_delay'
        deadline = time.time() + delay_s
        while time.time() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                return None
            time.sleep(min(0.05, max(deadline - time.time(), 0.0)))
    reference_norm = tactile_manager.total_norm()
    tactile_manager.set_release_reference(reference_norm)
    print_tactile_frame_log(tactile_manager, stage='release_reference', stage_time_s=0.0)
    print(f'[Tactile] Release reference captured: norm={reference_norm:.3f}')
    return reference_norm

def execute_tactile_release_descent(controller, tactile_manager, fixed_orientation_base, start_pose_mm, args, cancel_event=None, on_state_read=None):
    """Descend in -Z until tactile release trigger or the configured lower bound."""
    start_x, start_y, start_z = [float(v) for v in start_pose_mm[:3]]
    requested_min_z_mm = max(HOME_PLACE_MIN_Z_MM, float(getattr(args, 'tactile_release_descent_min_z_mm', HOME_PLACE_MIN_Z_MM)))
    min_z_mm = max(requested_min_z_mm,float(args.workspace_z[0]))
    if min_z_mm>float(args.workspace_z[1]):
        raise ValueError('Tactile release minimum Z exceeds workspace upper bound')
    poll_dt = max(0.0, float(getattr(args, 'tactile_release_descent_poll_dt_s', DEFAULT_TACTILE_RELEASE_DESCENT_POLL_DT_S)))
    delta_threshold = max(0.0, float(getattr(args, 'tactile_release_delta_threshold', DEFAULT_TACTILE_RELEASE_DELTA_THRESHOLD)))
    release_stop_timing_debug = bool(getattr(args, 'tactile_release_stop_timing_debug', DEFAULT_TACTILE_RELEASE_STOP_TIMING_DEBUG))
    release_stop_speed_threshold_mps = max(0.0, float(getattr(args, 'tactile_release_stop_speed_threshold_mps', DEFAULT_TACTILE_RELEASE_STOP_SPEED_THRESHOLD_MPS)))
    release_stop_monitor_timeout_s = max(0.0, float(getattr(args, 'tactile_release_stop_monitor_timeout_s', DEFAULT_TACTILE_RELEASE_STOP_MONITOR_TIMEOUT_S)))
    release_stop_monitor_poll_dt_s = max(0.0, float(getattr(args, 'tactile_release_stop_monitor_poll_dt_s', DEFAULT_TACTILE_RELEASE_STOP_MONITOR_POLL_DT_S)))
    tolerance_mm = max(0.0, float(getattr(args, 'position_tolerance_m', 0.0)) * 1000.0)
    current_pose_mm = (start_x, start_y, max(start_z, min_z_mm))
    descent_start_unix = time.time()
    descent_start_perf = time.perf_counter()
    deadline = descent_start_unix + max(0.0, float(getattr(args, 'move_timeout_s', 0.0)))
    tactile_manager.release_status = 'release_descending'
    if cancel_event is not None and cancel_event.is_set():
        safe_stop_xarm(controller)
        tactile_manager.release_status = 'release_cancelled'
        return {'triggered': False, 'timed_out': False, 'reached_min_z': False, 'release_pose_mm': current_pose_mm}
    print(f'[Tactile] Release descent start: start=({start_x:.1f}, {start_y:.1f}, {start_z:.1f}), target_z={min_z_mm:.1f}, delta_threshold={delta_threshold:.3f}.')
    if min_z_mm>requested_min_z_mm:
        print(f'[INFO] Release minimum Z limited by workspace: requested={requested_min_z_mm:.1f} mm, effective={min_z_mm:.1f} mm.')
    try:
        send_robot_command(controller, ROBOT_CMD_MOVE_TO_POSITION, target_position_base=mm_to_m_tuple([start_x, start_y, min_z_mm]), fixed_orientation_base=fixed_orientation_base, gripper_action=GRIPPER_HOLD, source_mode='tactile_release_descent')
    except Exception:
        tactile_manager.release_status='release_descent_command_failed'
        raise

    def fmt_debug_value(value, precision=6, suffix=''):
        if value is None:
            return '-'
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(value_float):
            return '-'
        return f'{value_float:.{precision}f}{suffix}'

    def log_release_stop_timing(event, perf_s=None, unix_s=None, **values):
        if not release_stop_timing_debug:
            return
        now_perf = time.perf_counter() if perf_s is None else float(perf_s)
        now_unix = time.time() if unix_s is None else float(unix_s)
        elapsed_s = now_perf - descent_start_perf
        fields = [f'event={event}', f't_rel_s={elapsed_s:.6f}', f'unix_s={now_unix:.6f}']
        for key, value in values.items():
            fields.append(f'{key}={value}')
        print('[TactileTiming] ' + ' '.join(fields), flush=True)

    def monitor_release_stop(trigger_info, stop_issue_perf, stop_return_perf):
        if not release_stop_timing_debug:
            return
        trigger_perf = trigger_info.get('trigger_perf_s')
        tcp_stop_perf = None
        steady_perf = None
        last_linear_speed = None
        last_angular_speed = None
        last_is_steady = None
        read_error = None
        deadline_perf = time.perf_counter() + release_stop_monitor_timeout_s
        while time.perf_counter() <= deadline_perf:
            state_read_unix = time.time()
            try:
                state = controller.read_robot_state(now_timestamp=state_read_unix)
                if on_state_read is not None:
                    on_state_read(state)
                linear_speed, angular_speed = compute_tcp_speed_norms(state)
                last_linear_speed = linear_speed
                last_angular_speed = angular_speed
            except Exception as exc:
                read_error = repr(exc)
                linear_speed = None
                angular_speed = None
            is_steady = read_xarm_is_steady(controller)
            observed_perf = time.perf_counter()
            observed_unix = time.time()
            if is_steady is not None:
                last_is_steady = is_steady
            if tcp_stop_perf is None and linear_speed is not None and (linear_speed <= release_stop_speed_threshold_mps):
                tcp_stop_perf = observed_perf
                log_release_stop_timing('tcp_speed_stopped', perf_s=observed_perf, unix_s=observed_unix, linear_speed_mps=fmt_debug_value(linear_speed), angular_speed=fmt_debug_value(angular_speed), threshold_mps=fmt_debug_value(release_stop_speed_threshold_mps), command_to_tcp_stop_s=fmt_debug_value(tcp_stop_perf - stop_issue_perf), trigger_to_tcp_stop_s=fmt_debug_value(tcp_stop_perf - trigger_perf))
            if steady_perf is None and is_steady is True:
                steady_perf = observed_perf
                log_release_stop_timing('isSteady_true', perf_s=observed_perf, unix_s=observed_unix, linear_speed_mps=fmt_debug_value(linear_speed), command_to_isSteady_s=fmt_debug_value(steady_perf - stop_issue_perf), trigger_to_isSteady_s=fmt_debug_value(steady_perf - trigger_perf))
            if tcp_stop_perf is not None and steady_perf is not None:
                break
            if release_stop_monitor_poll_dt_s <= 0.0:
                time.sleep(0.0)
            else:
                time.sleep(release_stop_monitor_poll_dt_s)
        if tcp_stop_perf is None:
            log_release_stop_timing('tcp_speed_stop_timeout', linear_speed_mps=fmt_debug_value(last_linear_speed), angular_speed=fmt_debug_value(last_angular_speed), threshold_mps=fmt_debug_value(release_stop_speed_threshold_mps), timeout_s=fmt_debug_value(release_stop_monitor_timeout_s), read_error=read_error if read_error is not None else '-')
        if steady_perf is None:
            log_release_stop_timing('isSteady_timeout', isSteady=last_is_steady if last_is_steady is not None else '-', timeout_s=fmt_debug_value(release_stop_monitor_timeout_s))
        log_release_stop_timing('actual_stop_summary', trigger_to_command_s=fmt_debug_value(stop_issue_perf - trigger_perf), command_call_s=fmt_debug_value(stop_return_perf - stop_issue_perf), command_to_tcp_stop_s=fmt_debug_value(None if tcp_stop_perf is None else tcp_stop_perf - stop_issue_perf), command_to_isSteady_s=fmt_debug_value(None if steady_perf is None else steady_perf - stop_issue_perf), trigger_to_tcp_stop_s=fmt_debug_value(None if tcp_stop_perf is None else tcp_stop_perf - trigger_perf), trigger_to_isSteady_s=fmt_debug_value(None if steady_perf is None else steady_perf - trigger_perf))

    def stop_for_release_trigger(trigger_info):
        stop_issue_perf = time.perf_counter()
        stop_issue_unix = time.time()
        log_release_stop_timing('stop_command_issue', perf_s=stop_issue_perf, unix_s=stop_issue_unix, trigger_to_command_s=fmt_debug_value(stop_issue_perf - trigger_info.get('trigger_perf_s')))
        safe_stop_xarm(controller)
        stop_return_perf = time.perf_counter()
        stop_return_unix = time.time()
        log_release_stop_timing('stop_command_return', perf_s=stop_return_perf, unix_s=stop_return_unix, command_call_s=fmt_debug_value(stop_return_perf - stop_issue_perf), trigger_to_return_s=fmt_debug_value(stop_return_perf - trigger_info.get('trigger_perf_s')))
        monitor_release_stop(trigger_info, stop_issue_perf, stop_return_perf)

    def read_current_pose_mm():
        state = controller.read_robot_state(now_timestamp=time.time())
        if on_state_read is not None:
            on_state_read(state)
        if not state.valid:
            return None
        pose = state.actual_tcp_pose_base
        if pose is None:
            return None
        pose_mm = meters_to_mm(pose[:3])
        return tuple((float(v) for v in pose_mm[:3]))

    def check_trigger(pose_mm):
        current_norm = tactile_manager.total_norm()
        delta_norm = tactile_manager.update_release_delta(current_norm)
        print_tactile_frame_log(tactile_manager, stage='release_descent', stage_time_s=time.perf_counter() - descent_start_perf, z_mm=f'{pose_mm[2]:.1f}')
        if delta_norm is not None and delta_norm > delta_threshold:
            trigger_perf = time.perf_counter()
            trigger_unix = time.time()
            sample_perf = getattr(tactile_manager, 'last_sample_perf_s', np.nan)
            sample_unix = getattr(tactile_manager, 'last_sample_unix_s', np.nan)
            sample_age_s = None
            if sample_perf is not None:
                try:
                    sample_perf_float = float(sample_perf)
                    if np.isfinite(sample_perf_float):
                        sample_age_s = trigger_perf - sample_perf_float
                except (TypeError, ValueError):
                    sample_age_s = None
            tactile_manager.release_status = 'release_triggered'
            print(f'[Tactile] Release trigger detected during descent: norm={current_norm:.3f}, ref={tactile_manager.release_reference_norm:.3f}, delta={delta_norm:.3f} > {delta_threshold:.3f}.')
            log_release_stop_timing('trigger_detected', perf_s=trigger_perf, unix_s=trigger_unix, descent_s=fmt_debug_value(trigger_perf - descent_start_perf), norm=fmt_debug_value(current_norm, precision=3), ref=fmt_debug_value(tactile_manager.release_reference_norm, precision=3), delta=fmt_debug_value(delta_norm, precision=3), threshold=fmt_debug_value(delta_threshold, precision=3), z_mm=fmt_debug_value(pose_mm[2], precision=1), tactile_sample_unix_s=fmt_debug_value(sample_unix), tactile_sample_age_s=fmt_debug_value(sample_age_s))
            return {'trigger_perf_s': trigger_perf, 'trigger_unix_s': trigger_unix, 'current_norm': float(current_norm), 'reference_norm': tactile_manager.release_reference_norm, 'delta_norm': float(delta_norm), 'delta_threshold': float(delta_threshold), 'pose_mm': tuple((float(v) for v in pose_mm[:3])), 'sample_perf_s': sample_perf, 'sample_unix_s': sample_unix, 'sample_age_s': sample_age_s}
        return None
    while True:
        if cancel_event is not None and cancel_event.is_set():
            safe_stop_xarm(controller)
            tactile_manager.release_status = 'release_cancelled'
            return {'triggered': False, 'timed_out': False, 'reached_min_z': False, 'release_pose_mm': current_pose_mm}
        pose_mm = read_current_pose_mm()
        if pose_mm is None:
            tactile_manager.release_status = 'release_descent_pose_missing'
            safe_stop_xarm(controller)
            print('[WARN] Tactile release descent has invalid TCP feedback; keeping gripper closed.')
            return {'triggered': False, 'timed_out': False, 'reached_min_z': False, 'failed':True,'release_pose_mm': current_pose_mm}
        current_pose_mm = pose_mm
        trigger_info = check_trigger(current_pose_mm)
        if trigger_info is not None:
            stop_for_release_trigger(trigger_info)
            return {'triggered': True, 'timed_out': False, 'reached_min_z': False, 'release_pose_mm': current_pose_mm}
        if current_pose_mm[2] <= min_z_mm + tolerance_mm:
            tactile_manager.release_status = 'release_min_z_open'
            safe_stop_xarm(controller)
            print(f'[Tactile] Release descent reached min z without trigger; opening gripper. z={current_pose_mm[2]:.1f}, min_z={min_z_mm:.1f}.')
            return {'triggered': False, 'timed_out': False, 'reached_min_z': True, 'release_pose_mm': current_pose_mm}
        if time.time() >= deadline:
            tactile_manager.release_status = 'release_descent_move_timeout'
            safe_stop_xarm(controller)
            print(f'[WARN] Tactile release descent move timed out; opening gripper at current pose. z={current_pose_mm[2]:.1f}, min_z={min_z_mm:.1f}.')
            return {'triggered': False, 'timed_out': True, 'reached_min_z': False, 'release_pose_mm': current_pose_mm}
        if poll_dt > 0.0:
            time.sleep(poll_dt)

def reset_tactile_baseline_after_open(tactile_manager, delay_s, cancel_event=None):
    if tactile_manager is None or not bool(getattr(tactile_manager, 'enabled', False)):
        return
    if cancel_event is not None and cancel_event.is_set():
        return
    delay_s = max(0.0, float(delay_s))
    if delay_s <= 0.0:
        return
    tactile_manager.release_status = 'baseline_reset_delay'
    deadline = time.time() + delay_s
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            return
        time.sleep(min(0.05, max(deadline - time.time(), 0.0)))
    if tactile_manager.reset_baseline():
        tactile_manager.release_status = 'baseline_reset'

def save_grasp_offset(controller, shared_state, on_state_read=None):
    """Store the object-to-tool offset after a verified grasp."""
    snap = shared_state.get_snapshot()
    obj_xyz = snap['latest_object_xyz_mm']
    if obj_xyz is None:
        print('[WARN] Cannot save grasp offset: no latest object xyz.')
        return False
    state = controller.read_robot_state(now_timestamp=time.time())
    if on_state_read is not None:
        on_state_read(state)
    cur_pose = state.actual_tcp_pose_base
    if cur_pose is None:
        print('[WARN] Cannot read EEF pose for grasp offset.')
        return False
    eef_xyz = meters_to_mm(cur_pose[:3])
    grasp_offset_xyz = obj_xyz - eef_xyz
    with shared_state.lock:
        shared_state.grasp_offset_xyz_mm = grasp_offset_xyz
        shared_state.grasp_closed = True
    shared_state.set_task_state('GRASPED', reset_prediction=True, reset_arm=True)
    print(f'[INFO] grasp_offset_xyz_mm saved: {grasp_offset_xyz}')
    return True

def compute_place_target(shared_state, place_object_xy_mm=DEFAULT_PLACE_OBJECT_XY_MM, *, apply_grasp_offset_xy=True):
    """Compute the fixed place TCP target, optionally compensating grasp XY offset."""
    fixed_xyz = getattr(shared_state.args, 'place_object_xyz_mm', None)
    if fixed_xyz is not None:
        target = np.asarray(fixed_xyz, dtype=np.float64).reshape(3)
        if not np.isfinite(target).all():
            raise ValueError('Fixed place XYZ must be finite')
        for axis, value in zip('xyz', target):
            low, high = getattr(shared_state.args, f'workspace_{axis}')
            if not low <= value <= high:
                raise ValueError(f'Fixed place {axis.upper()} outside workspace')
        return target, {'used_fallback': False, 'fallback_reason': 'none', 'grasp_z_median_mm': None,
                        'template_bottom_z_median_mm': None, 'raw_place_z_mm': float(target[2]),
                        'grasp_offset_xy_applied': False, 'applied_grasp_offset_x_mm': 0.0,
                        'applied_grasp_offset_y_mm': 0.0}
    place_object_xy = np.asarray(place_object_xy_mm, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(place_object_xy)):
        raise ValueError('place_object_xy_mm must contain finite values')
    with shared_state.lock:
        grasp_offset = None if shared_state.grasp_offset_xyz_mm is None else shared_state.grasp_offset_xyz_mm.copy()
        frozen_place_z_mm = shared_state.frozen_place_z_mm
        frozen_place_z_raw_mm = shared_state.frozen_place_z_raw_mm
        frozen_place_z_grasp_median_mm = shared_state.frozen_place_z_grasp_median_mm
        frozen_place_z_template_bottom_median_mm = shared_state.frozen_place_z_template_bottom_median_mm
    if apply_grasp_offset_xy and grasp_offset is None:
        return (None, {'used_fallback': False, 'fallback_reason': 'no_grasp_offset', 'grasp_z_median_mm': frozen_place_z_grasp_median_mm, 'template_bottom_z_median_mm': frozen_place_z_template_bottom_median_mm, 'raw_place_z_mm': frozen_place_z_raw_mm})
    if frozen_place_z_mm is None or not np.isfinite(float(frozen_place_z_mm)):
        return (None, {'used_fallback': False, 'fallback_reason': 'no_frozen_place_z', 'grasp_z_median_mm': frozen_place_z_grasp_median_mm, 'template_bottom_z_median_mm': frozen_place_z_template_bottom_median_mm, 'raw_place_z_mm': frozen_place_z_raw_mm})
    if grasp_offset is not None:
        grasp_offset = np.asarray(grasp_offset, dtype=np.float64).reshape(3)
    if apply_grasp_offset_xy and (not np.all(np.isfinite(grasp_offset))):
        return (None, {'used_fallback': False, 'fallback_reason': 'invalid_grasp_offset', 'grasp_z_median_mm': frozen_place_z_grasp_median_mm, 'template_bottom_z_median_mm': frozen_place_z_template_bottom_median_mm, 'raw_place_z_mm': frozen_place_z_raw_mm})
    applied_grasp_offset_xy = grasp_offset[:2] if apply_grasp_offset_xy else np.zeros(2, dtype=np.float64)
    target_xyz = np.asarray([place_object_xy[0] - applied_grasp_offset_xy[0], place_object_xy[1] - applied_grasp_offset_xy[1], max(float(frozen_place_z_mm), HOME_PLACE_MIN_Z_MM)], dtype=np.float64)
    debug = {'used_fallback': False, 'fallback_reason': 'none', 'grasp_z_median_mm': frozen_place_z_grasp_median_mm, 'template_bottom_z_median_mm': frozen_place_z_template_bottom_median_mm, 'raw_place_z_mm': float(frozen_place_z_mm) if frozen_place_z_raw_mm is None else float(frozen_place_z_raw_mm), 'place_object_x_mm': float(place_object_xy[0]), 'place_object_y_mm': float(place_object_xy[1]), 'grasp_offset_xy_applied': bool(apply_grasp_offset_xy), 'applied_grasp_offset_x_mm': float(applied_grasp_offset_xy[0]), 'applied_grasp_offset_y_mm': float(applied_grasp_offset_xy[1])}
    return (target_xyz, debug)

def compute_pre_release_descend_target_mm(place_x, place_y, place_z, args):
    """Resolve the optional descend target used immediately before gripper open."""
    enabled = bool(getattr(args, 'pre_release_descend_before_open', False))
    descend_mm = max(0.0, float(getattr(args, 'pre_release_descend_mm', RELEASE_PARAMETER_MM)))
    place_x = float(place_x)
    place_y = float(place_y)
    place_z = float(place_z)
    if not enabled or descend_mm <= 1e-09:
        return ((place_x, place_y, place_z), {'enabled': False, 'descend_mm': descend_mm, 'unclamped_z_mm': place_z, 'target_z_mm': place_z, 'home_guard_applied': False})
    unclamped_z = place_z - descend_mm
    min_release_z = HOME_PLACE_MIN_Z_MM + PRE_RELEASE_MIN_Z_EPSILON_MM
    workspace_z = args.workspace_z
    workspace_z_low = float(workspace_z[0])
    workspace_z_high = float(workspace_z[1])
    guarded_low = max(min_release_z, workspace_z_low)
    target_z = clamp_value(unclamped_z, guarded_low, workspace_z_high)
    if target_z <= HOME_PLACE_MIN_Z_MM:
        target_z = min_release_z
    target_x = clamp_value(place_x, args.workspace_x[0], args.workspace_x[1])
    target_y = clamp_value(place_y, args.workspace_y[0], args.workspace_y[1])
    home_guard_applied = unclamped_z < min_release_z or target_z <= HOME_PLACE_MIN_Z_MM
    return ((target_x, target_y, target_z), {'enabled': True, 'descend_mm': descend_mm, 'unclamped_z_mm': unclamped_z, 'target_z_mm': target_z, 'home_guard_applied': bool(home_guard_applied)})

def execute_return_and_place(controller, shared_state, args, metadata_recorder=None, tactile_manager=None, cancel_event=None, on_state_read=None):
    """Run the post-grasp return, release, backoff, and HOME sequence."""
    if cancel_event is not None and cancel_event.is_set():
        safe_stop_xarm(controller)
        return False
    place_object_xy_mm = tuple((float(value) for value in getattr(args, 'place_object_xy_mm', DEFAULT_PLACE_OBJECT_XY_MM)))
    target_eef_xyz, place_target_debug = compute_place_target(shared_state, place_object_xy_mm=place_object_xy_mm, apply_grasp_offset_xy=bool(getattr(args, 'apply_place_grasp_offset_xy', True)))
    if target_eef_xyz is None:
        print(f"[WARN] Cannot compute fixed place target: reason={place_target_debug.get('fallback_reason', 'unknown')}")
        return False
    snap = shared_state.get_snapshot()
    fixed_orientation_base = snap['fixed_orientation_base']
    initial_pose_base = snap['initial_pose_base']
    if fixed_orientation_base is None:
        print('[WARN] No fixed orientation.')
        return False
    tactile_enabled = tactile_manager is not None and bool(getattr(tactile_manager, 'enabled', False))

    fixed_place = getattr(args, 'place_object_xyz_mm', None) is not None

    def move_with_optional_cancel(target_position_base, fixed_orientation_base, *, timeout_s, tolerance_m, source_mode):
        kwargs = {'timeout_s': timeout_s, 'tolerance_m': tolerance_m, 'source_mode': source_mode}
        if cancel_event is not None:
            kwargs['cancel_event'] = cancel_event
        if on_state_read is not None:
            kwargs['on_state_read'] = on_state_read
        return move_robot_and_wait(controller, target_position_base, fixed_orientation_base, **kwargs)
    target_x, target_y, target_z = target_eef_xyz
    target_x, target_y, target_z = clamp_pose_mm(target_x, target_y, target_z, args)
    hover_z = target_z + HOVER_Z_OFFSET_MM
    hover_x, hover_y, hover_z = clamp_pose_mm(target_x, target_y, hover_z, args)
    place_z = target_z + DESCEND_EXTRA_MM
    place_x, place_y, place_z = clamp_pose_mm(target_x, target_y, place_z, args)
    if fixed_place:
        place_x, place_y, place_z = target_eef_xyz
        descend = float(getattr(args, 'place_descend_mm', 0.0))
        release_z = float(place_z) - descend
        if not np.isfinite(descend) or descend < 0 or not args.workspace_z[0] <= release_z <= args.workspace_z[1]:
            raise ValueError('Invalid fixed box descent target Z')
        release_pose_mm, pre_release_debug = (place_x, place_y, release_z), {'enabled': False}
    else:
        release_pose_mm, pre_release_debug = compute_pre_release_descend_target_mm(place_x, place_y, place_z, args)
    release_x, release_y, release_z = release_pose_mm
    grasp_z_med_text = '-' if place_target_debug['grasp_z_median_mm'] is None else f"{float(place_target_debug['grasp_z_median_mm']):.1f}"
    template_bottom_z_med_text = '-' if place_target_debug['template_bottom_z_median_mm'] is None else f"{float(place_target_debug['template_bottom_z_median_mm']):.1f}"
    raw_place_z_text = '-' if place_target_debug['raw_place_z_mm'] is None else f"{float(place_target_debug['raw_place_z_mm']):.1f}"
    print(f'[INFO] RETURN hover target: ({hover_x:.1f}, {hover_y:.1f}, {hover_z:.1f})')
    print(f'[INFO] PLACE target: ({place_x:.1f}, {place_y:.1f}, {place_z:.1f})')
    print(f"[INFO] PLACE XY mode: grasp_offset_applied={place_target_debug['grasp_offset_xy_applied']} configured_object_xy=({place_object_xy_mm[0]:.1f}, {place_object_xy_mm[1]:.1f}) applied_offset_xy=({place_target_debug['applied_grasp_offset_x_mm']:.1f}, {place_target_debug['applied_grasp_offset_y_mm']:.1f})")
    print(f"[INFO] PLACE z debug: grasp_z_med={grasp_z_med_text} template_bottom_z_med={template_bottom_z_med_text} raw_place_z={raw_place_z_text} fallback={place_target_debug['used_fallback']}")
    if fixed_place:
        print(f'[INFO] Fixed box release TCP: ({release_x:.1f}, {release_y:.1f}, {release_z:.1f}) mm; open after arrival.')
    elif pre_release_debug['enabled'] and (not tactile_enabled):
        print(f"[INFO] PRE-release descend target: ({release_x:.1f}, {release_y:.1f}, {release_z:.1f}), descend={pre_release_debug['descend_mm']:.1f} mm, unclamped_z={pre_release_debug['unclamped_z_mm']:.1f} mm, home_guard_applied={pre_release_debug['home_guard_applied']}")
    elif tactile_enabled:
        print(f"[INFO] Tactile release descent enabled. Ignoring fixed pre-release descend distance={pre_release_debug['descend_mm']:.1f} mm; min_z={float(getattr(args, 'tactile_release_descent_min_z_mm', HOME_PLACE_MIN_Z_MM)):.1f} mm.")
    move_sequence = [('return_hover', [hover_x, hover_y, hover_z]), ('return_place', [place_x, place_y, place_z])]
    if fixed_place:
        move_sequence = [('fixed_box_place', [place_x, place_y, place_z])]
        if descend > 0:
            move_sequence.append(('fixed_box_descend', [release_x, release_y, release_z]))
    for source_mode, pose_mm in move_sequence:
        if cancel_event is not None and cancel_event.is_set():
            safe_stop_xarm(controller)
            return False
        ok = move_with_optional_cancel(mm_to_m_tuple(pose_mm), fixed_orientation_base, timeout_s=args.move_timeout_s, tolerance_m=args.position_tolerance_m, source_mode=source_mode)
        if not ok:
            print(f'[WARN] Move timed out during {source_mode}.')
            return False
    if tactile_enabled and not fixed_place:
        capture_tactile_release_reference(tactile_manager, delay_s=getattr(args, 'tactile_release_ref_delay_s', DEFAULT_TACTILE_RELEASE_REF_DELAY_S), cancel_event=cancel_event)
        if cancel_event is not None and cancel_event.is_set():
            safe_stop_xarm(controller)
            return False
        descent_kwargs = {}
        if cancel_event is not None:
            descent_kwargs['cancel_event'] = cancel_event
        if on_state_read is not None:
            descent_kwargs['on_state_read'] = on_state_read
        descent_result = execute_tactile_release_descent(controller, tactile_manager, fixed_orientation_base, [place_x, place_y, place_z], args, **descent_kwargs)
        if descent_result.get('failed',False):
            return False
        release_x, release_y, release_z = descent_result['release_pose_mm']
    elif pre_release_debug['enabled'] and not fixed_place:
        ok = move_with_optional_cancel(mm_to_m_tuple([release_x, release_y, release_z]), fixed_orientation_base, timeout_s=args.move_timeout_s, tolerance_m=args.position_tolerance_m, source_mode='pre_release_descend_before_open')
        if not ok:
            print('[WARN] Pre-release descend move timed out.')
            return False
    if not execute_gripper_open(controller, dwell_s=args.gripper_release_dwell_s, metadata_recorder=metadata_recorder, cancel_event=cancel_event):
        return False
    if cancel_event is not None and cancel_event.is_set():
        return False
    if tactile_enabled:
        reset_tactile_baseline_after_open(tactile_manager, getattr(args, 'tactile_auto_baseline_reset_after_open_s', DEFAULT_TACTILE_AUTO_BASELINE_RESET_AFTER_OPEN_S), cancel_event=cancel_event)
    post_release_offset = 0.0 if fixed_place else float(args.post_release_z_offset_mm)
    post_release_z = release_z if fixed_place else max(release_z + post_release_offset, HOME_PLACE_MIN_Z_MM)
    post_release_x, post_release_y, post_release_z = clamp_pose_mm(release_x, release_y, post_release_z, args)
    if abs(post_release_offset) > 1e-06:
        print(f'[INFO] POST-release Z target: ({post_release_x:.1f}, {post_release_y:.1f}, {post_release_z:.1f}), offset={float(args.post_release_z_offset_mm):.1f} mm')
        ok = move_with_optional_cancel(mm_to_m_tuple([post_release_x, post_release_y, post_release_z]), fixed_orientation_base, timeout_s=args.move_timeout_s, tolerance_m=args.position_tolerance_m, source_mode='post_release_z_move')
        if not ok:
            print('[WARN] Post-release Z move timed out.')
            return False
    backoff_x = place_x - BACKOFF_X_MM
    backoff_x, backoff_y, backoff_z = clamp_pose_mm(backoff_x, place_y, post_release_z, args)
    ok = move_with_optional_cancel(mm_to_m_tuple([backoff_x, backoff_y, backoff_z]), fixed_orientation_base, timeout_s=args.move_timeout_s, tolerance_m=args.position_tolerance_m, source_mode='return_backoff')
    if not ok:
        print('[WARN] Back off move timed out.')
        return False
    if bool(getattr(args, 'post_backoff_stop_check_enabled', DEFAULT_POST_BACKOFF_STOP_CHECK_ENABLED)):
        stop_confirmed = wait_until_robot_stopped(controller, timeout_s=getattr(args, 'post_backoff_stop_timeout_s', DEFAULT_POST_BACKOFF_STOP_TIMEOUT_S), speed_threshold_mps=getattr(args, 'post_backoff_stop_speed_threshold_mps', DEFAULT_POST_BACKOFF_STOP_SPEED_THRESHOLD_MPS), poll_dt=getattr(args, 'post_backoff_stop_poll_dt_s', DEFAULT_POST_BACKOFF_STOP_POLL_DT_S), cancel_event=cancel_event, on_state_read=on_state_read, source_mode='post_backoff_stop_check')
        if cancel_event is not None and cancel_event.is_set():
            return False
        if stop_confirmed:
            print('[INFO] POST-backoff stop confirmed before HOME moveJ.')
        else:
            message = '[WARN] POST-backoff stop was not confirmed before HOME moveJ.'
            if bool(getattr(args, 'post_backoff_stop_require_confirmed', DEFAULT_POST_BACKOFF_STOP_REQUIRE_CONFIRMED)):
                print(message + ' Aborting return sequence because confirmation is required.')
                return False
            print(message + ' Continuing with HOME moveJ.')
    try:
        home_kwargs = {}
        if cancel_event is not None:
            home_kwargs['cancel_event'] = cancel_event
        if on_state_read is not None:
            home_kwargs['on_state_read'] = on_state_read
        home_ok = move_robot_to_home_pose(controller, args, **home_kwargs)
        if home_ok is False:
            return False
    except Exception as exc:
        print(f'[WARN] Return to HOME joints failed: {exc}')
        if initial_pose_base is None:
            return False
        initial_target = tuple((float(v) for v in initial_pose_base[:3]))
        initial_orientation = tuple((float(v) for v in initial_pose_base[3:6]))
        ok = move_with_optional_cancel(initial_target, initial_orientation, timeout_s=args.move_timeout_s, tolerance_m=args.position_tolerance_m, source_mode='return_initial_pose')
        if not ok:
            print('[WARN] Return to initial pose timed out.')
            return False
    if metadata_recorder is not None:
        if fixed_place:
            metadata_recorder.note_release_tcp_target((release_x, release_y, release_z))
        else:
            metadata_recorder.note_delivery_bottom_location((float(place_object_xy_mm[0]), float(place_object_xy_mm[1]), float(RELEASE_PARAMETER_MM)))
    shared_state.set_task_state('DONE', reset_prediction=True, reset_arm=True)
    print('[INFO] RETURN + PLACE done')
    return True

def build_fitted_merged_object(raw_merged_object, shape_fitting_state):
    """Expose the shape-fitted template cloud through the merger state interface."""
    if not shape_fitting_state.valid:
        return replace(raw_merged_object, object_detected=False, centroid_base=None, merged_point_count=0, merged_points_base=[], valid=False)
    fitted_points = np.asarray(shape_fitting_state.fitted_points_base, dtype=np.float32).reshape((-1, 3))
    return replace(raw_merged_object, object_detected=True, label=shape_fitting_state.label or raw_merged_object.label, centroid_base=shape_fitting_state.centroid_base, merged_point_count=int(len(fitted_points)), merged_points_base=[tuple((float(v) for v in point)) for point in fitted_points], valid=True)

def project_base_point_to_camera(point_base, intrinsics, t_cam_base, width: int, height: int):
    """Project one base-frame point into a camera image for overlay drawing."""
    if point_base is None or intrinsics is None:
        return None
    point = np.asarray(point_base, dtype=np.float32).reshape(3)
    point_h = np.concatenate([point, np.array([1.0], dtype=np.float32)], axis=0)
    point_cam = (t_cam_base @ point_h.reshape(4, 1)).reshape(-1)[:3]
    z = float(point_cam[2])
    if not np.isfinite(z) or z <= 1e-06:
        return None
    fx = float(intrinsics['fx'])
    fy = float(intrinsics['fy'])
    cx = float(intrinsics['cx'])
    cy = float(intrinsics['cy'])
    u = int(round(float(point_cam[0]) * fx / z + cx))
    v = int(round(float(point_cam[1]) * fy / z + cy))
    if u < 0 or u >= int(width) or v < 0 or (v >= int(height)):
        return None
    return (u, v)

def project_base_points_to_camera(points_base, intrinsics, t_cam_base, width: int, height: int, max_points: int=1500):
    """Project a sampled base-frame point cloud into camera pixels for preview."""
    if points_base is None or intrinsics is None:
        return np.empty((0, 2), dtype=np.int32)
    points = np.asarray(points_base, dtype=np.float32).reshape((-1, 3))
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32)
    if len(points) > max_points:
        stride = max(1, len(points) // max_points)
        points = points[::stride]
    ones = np.ones((len(points), 1), dtype=np.float32)
    points_h = np.concatenate([points, ones], axis=1)
    points_cam = (t_cam_base @ points_h.T).T[:, :3]
    zs = points_cam[:, 2]
    valid = np.isfinite(zs) & (zs > 1e-06)
    if not np.any(valid):
        return np.empty((0, 2), dtype=np.int32)
    points_cam = points_cam[valid]
    zs = zs[valid]
    fx = float(intrinsics['fx'])
    fy = float(intrinsics['fy'])
    cx = float(intrinsics['cx'])
    cy = float(intrinsics['cy'])
    us = np.round(points_cam[:, 0] * fx / zs + cx).astype(np.int32)
    vs = np.round(points_cam[:, 1] * fy / zs + cy).astype(np.int32)
    in_bounds = (us >= 0) & (us < int(width)) & (vs >= 0) & (vs < int(height))
    if not np.any(in_bounds):
        return np.empty((0, 2), dtype=np.int32)
    return np.stack([us[in_bounds], vs[in_bounds]], axis=1)

def draw_projected_template_overlay(image_bgr, points_base, intrinsics, t_cam_base, *, max_points: int=1800, color_bgr=(0, 0, 255)):
    """Draw a fitted base-frame template cloud into one camera preview."""
    template_pixels = project_base_points_to_camera(points_base, intrinsics, t_cam_base, image_bgr.shape[1], image_bgr.shape[0], max_points=max_points)
    if len(template_pixels) == 0:
        return image_bgr
    point_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
    point_mask[template_pixels[:, 1], template_pixels[:, 0]] = 255
    point_mask = cv.dilate(point_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    image_bgr[point_mask > 0] = np.array(color_bgr, dtype=np.uint8)
    return image_bgr

def _build_silhouette_observation(camera_id, frame_bundle, object_worker, t_base_cam):
    """Package the latest per-camera segmentation mask for shape fitting rerank."""
    object_debug = getattr(object_worker, 'last_debug', None)
    mask = None if object_debug is None else getattr(object_debug, 'combined_mask', None)
    if mask is None:
        return None
    mask_array = np.asarray(mask)
    if mask_array.size == 0 or not np.any(mask_array):
        return None
    intrinsics = getattr(frame_bundle, 'intrinsics', None) or {}
    required_keys = ('fx', 'fy', 'cx', 'cy')
    if any((key not in intrinsics for key in required_keys)):
        return None
    image_shape = tuple((int(value) for value in frame_bundle.color_image.shape[:2]))
    return SilhouetteObservation(camera_id=int(camera_id), mask=mask_array, intrinsics=dict(intrinsics), t_base_cam=np.asarray(t_base_cam, dtype=np.float32), image_shape=(image_shape[0], image_shape[1]), weight=1.0)

def build_silhouette_observations(snapshot, pipeline):
    """Build cam0/cam1 silhouette observations when masks and calibration are available."""
    transform_chain = pipeline.get('transform_chain')
    if transform_chain is None:
        return []
    observations = []
    cam0_observation = _build_silhouette_observation(0, snapshot.cam0, pipeline.get('object_worker_cam0'), transform_chain.t_base_cam0)
    if cam0_observation is not None:
        observations.append(cam0_observation)
    cam1_observation = _build_silhouette_observation(1, snapshot.cam1, pipeline.get('object_worker_cam1'), transform_chain.t_base_cam1)
    if cam1_observation is not None:
        observations.append(cam1_observation)
    return observations

def choose_point(primary, fallback=None):
    return primary if primary is not None else fallback

def offset_point_base_mm(point_base, *, x_mm=0.0, y_mm=0.0, z_mm=0.0):
    if point_base is None:
        return None
    point = np.asarray(point_base, dtype=np.float32).reshape(3).copy()
    point[0] += float(x_mm) / 1000.0
    point[1] += float(y_mm) / 1000.0
    point[2] += float(z_mm) / 1000.0
    return tuple((float(v) for v in point))

def append_debug_3d_frame(debug_3d_recorder, *, snapshot, current_time, loop_perf, record_elapsed_s, current_task_epoch, pipeline, selected_hand, merged_object, shape_fitting_state, object_point_base, grasp_point_base, eef_pose_base, measurement_source, tactile_manager=None):
    """Record one synchronized perception/debug frame for later 3D inspection."""
    tactile_snapshot = None
    if tactile_manager is not None and bool(getattr(tactile_manager, 'enabled', False)):
        try:
            if hasattr(tactile_manager, 'snapshot'):
                tactile_snapshot = tactile_manager.snapshot(refresh=True)
            else:
                if hasattr(tactile_manager, 'total_norm'):
                    tactile_manager.total_norm()
                values = np.asarray(getattr(tactile_manager, 'latest', []), dtype=np.float32).flatten()
                tactile_snapshot = {'values': values, 'num_mags': int(getattr(tactile_manager, 'num_mags', max(values.size // 3, 0))), 'total_norm': float(getattr(tactile_manager, 'latest_norm', float(np.linalg.norm(values)))), 'release_ref_norm': getattr(tactile_manager, 'release_reference_norm', None), 'release_delta_norm': getattr(tactile_manager, 'release_delta_norm', None), 'status': str(getattr(tactile_manager, 'release_status', getattr(tactile_manager, 'status', ''))), 'error': getattr(tactile_manager, 'last_error', None), 'timestamp_perf_s': float(getattr(tactile_manager, 'last_sample_perf_s', np.nan)), 'timestamp_unix_s': float(getattr(tactile_manager, 'last_sample_unix_s', np.nan))}
        except Exception as exc:
            tactile_snapshot = {'values': np.empty((0,), dtype=np.float32), 'num_mags': int(getattr(tactile_manager, 'num_mags', 0)), 'total_norm': np.nan, 'release_ref_norm': getattr(tactile_manager, 'release_reference_norm', None), 'release_delta_norm': getattr(tactile_manager, 'release_delta_norm', None), 'status': 'snapshot_error', 'error': str(exc)}
    debug_3d_recorder.append_frame(frame_index=int(snapshot.pair_index), timestamp_unix_s=current_time, timestamp_perf_s=loop_perf, record_elapsed_s=record_elapsed_s, task_epoch=current_task_epoch, selected_hand=selected_hand, hand_debug_cam0=getattr(pipeline['hand_worker_cam0'], 'last_debug', None), hand_debug_cam1=getattr(pipeline['hand_worker_cam1'], 'last_debug', None), raw_merged_object=merged_object, shape_fitting_state=shape_fitting_state, object_point_base=object_point_base, grasp_point_base=grasp_point_base, eef_pose_base=eef_pose_base, measurement_source=measurement_source, geometry_source='raw_point_cloud' if pipeline.get('ablation_mode') == ABLATION_SHAPE_FITTING else 'fitted_template', hand_selector_debug=getattr(pipeline['hand_selector'], 'last_debug', None), tactile_snapshot=tactile_snapshot, snapshot=snapshot if debug_3d_recorder.save_images else None)

def collect_runtime_profile_metrics(*, snapshot, pipeline, object_cam0, object_cam1, merged_object, shape_fitting_state, selected_hand, fusion_state, grasp_target, measurement_source, instant_fps, smoothed_fps):
    """Collect per-frame perception and timing metrics for runtime profiling."""
    object_debug_cam0 = getattr(pipeline['object_worker_cam0'], 'last_debug', None)
    object_debug_cam1 = getattr(pipeline['object_worker_cam1'], 'last_debug', None)
    shape_fit_debug = getattr(pipeline['shape_fitting_tracker'], 'last_debug', None)
    sam3d_runtime = pipeline.get('sam3d_live_runtime')
    sam3d_debug = None if sam3d_runtime is None else sam3d_runtime.debug()
    object_backend = str(pipeline.get('object_backend', 'sam3d'))
    cam0_object_infer_ms = getattr(object_debug_cam0, 'infer_ms', None)
    cam1_object_infer_ms = getattr(object_debug_cam1, 'infer_ms', None)
    return {'instant_fps': float(instant_fps), 'smoothed_fps': float(smoothed_fps), 'camera_pair_delta_ms': float(getattr(snapshot, 'timestamp_delta_ms', 0.0)), 'camera_pair_within_sync_tolerance': bool(getattr(snapshot, 'within_sync_tolerance', False)), 'cam0_object_infer_ms': cam0_object_infer_ms, 'cam1_object_infer_ms': cam1_object_infer_ms, 'cam0_object_detected': bool(getattr(object_cam0, 'object_detected', False)), 'cam1_object_detected': bool(getattr(object_cam1, 'object_detected', False)), 'cam0_object_points': int(getattr(object_cam0, 'point_count', 0)), 'cam1_object_points': int(getattr(object_cam1, 'point_count', 0)), 'merged_object_valid': bool(getattr(merged_object, 'valid', False)), 'merged_object_points': int(getattr(merged_object, 'merged_point_count', 0)), 'merged_object_label': getattr(merged_object, 'label', None), 'geometry_source': 'raw_point_cloud' if pipeline.get('ablation_mode') == ABLATION_SHAPE_FITTING else 'fitted_template', 'shape_fit_valid': bool(getattr(shape_fitting_state, 'valid', False)), 'shape_fit_initialized': bool(getattr(shape_fitting_state, 'initialized', False)), 'shape_fit_reason': getattr(shape_fitting_state, 'reason', None), 'shape_fit_icp_time_ms': getattr(shape_fit_debug, 'icp_time_ms', None), 'shape_fit_icp_fitness': getattr(shape_fit_debug, 'icp_fitness', None), 'shape_fit_icp_rmse': getattr(shape_fit_debug, 'icp_rmse', None), 'shape_fit_z_rotation_deg': getattr(shape_fit_debug, 'z_rotation_deg', None), 'shape_fit_roll_rotation_deg': getattr(shape_fit_debug, 'roll_rotation_deg', None), 'shape_fit_pitch_rotation_deg': getattr(shape_fit_debug, 'pitch_rotation_deg', None), 'shape_fit_rotation_search_mode': getattr(shape_fit_debug, 'rotation_search_mode', None), 'shape_fit_rotation_candidate_count': getattr(shape_fit_debug, 'rotation_candidate_count', 0), 'shape_fit_rotation_coarse_candidate_count': getattr(shape_fit_debug, 'rotation_coarse_candidate_count', 0), 'shape_fit_rotation_refine_candidate_count': getattr(shape_fit_debug, 'rotation_refine_candidate_count', 0), 'shape_fit_silhouette_enabled': getattr(shape_fit_debug, 'silhouette_enabled', False), 'shape_fit_silhouette_candidate_count': getattr(shape_fit_debug, 'silhouette_candidate_count', 0), 'shape_fit_silhouette_valid_camera_count': getattr(shape_fit_debug, 'silhouette_valid_camera_count', 0), 'shape_fit_silhouette_loss': getattr(shape_fit_debug, 'silhouette_loss', None), 'shape_fit_silhouette_outside_loss': getattr(shape_fit_debug, 'silhouette_outside_loss', None), 'shape_fit_silhouette_robust_3d_loss': getattr(shape_fit_debug, 'robust_3d_loss', None), 'shape_fit_rerank_changed_candidate': getattr(shape_fit_debug, 'rerank_changed_candidate', False), 'selected_hand_valid': bool(getattr(selected_hand, 'valid', False)), 'selected_hand_camera': getattr(selected_hand, 'selected_camera', None), 'fusion_object_fresh': bool(getattr(fusion_state, 'object_fresh', False)), 'fusion_hand_fresh': bool(getattr(fusion_state, 'hand_fresh', False)), 'fusion_hand_approach_latched': bool(getattr(fusion_state, 'hand_approach_latched', False)), 'grasp_target_valid': bool(getattr(grasp_target, 'valid', False)), 'measurement_source': str(measurement_source or 'none'), 'object_backend': object_backend, 'sam3d_dynamic_active': bool(getattr(sam3d_debug, 'active', False)), 'sam3d_gate_blocked': bool(getattr(sam3d_debug, 'gate_blocked', False)), 'sam3d_initialization_phase': getattr(sam3d_debug, 'initialization_phase', None), 'sam3d_scaling_valid_frames': getattr(sam3d_debug, 'scaling_valid_frames', 0), 'sam3d_scaling_required_frames': getattr(sam3d_debug, 'scaling_required_frames', 0), 'sam3d_silhouette_scale_frozen': bool(getattr(sam3d_debug, 'silhouette_scale_frozen', False))}

def save_runtime_profile(runtime_profiler, *, reason):
    if not getattr(runtime_profiler, 'enabled', False):
        return
    try:
        paths = runtime_profiler.save_current_session(reason=reason)
    except Exception as exc:
        print(f'[WARN] Runtime profile save failed: {exc}')
        return
    if paths:
        print(f"[INFO] Runtime profile saved: {paths.get('summary')}")

def discard_runtime_profile(runtime_profiler, *, reason):
    if not getattr(runtime_profiler, 'enabled', False):
        return
    runtime_profiler.discard_current_session(reason=reason)

def render_camera_mask_preview(snapshot, pipeline, object_worker, merged_object, selected_hand, fusion_state, display_grasp_point, *, camera_label):
    """Render a camera preview with object mask, fitted template, and key projected points."""
    frame_bundle = snapshot.cam0 if camera_label == 'cam0' else snapshot.cam1
    image_bgr = np.asarray(frame_bundle.color_image).copy()
    object_debug = getattr(object_worker, 'last_debug', None)
    combined_mask = getattr(object_debug, 'combined_mask', None)
    if combined_mask is not None:
        mask_bool = np.asarray(combined_mask, dtype=bool)
        if mask_bool.shape[:2] == image_bgr.shape[:2]:
            overlay = np.zeros_like(image_bgr, dtype=np.uint8)
            overlay[mask_bool] = np.array([255, 180, 0], dtype=np.uint8)
            image_bgr = cv.addWeighted(image_bgr, 1.0, overlay, 0.35, 0.0)
    camera_transform = pipeline['t_cam0_base'] if camera_label == 'cam0' else pipeline['t_cam1_base']
    image_bgr = draw_projected_template_overlay(image_bgr, getattr(merged_object, 'merged_points_base', None), frame_bundle.intrinsics, camera_transform, max_points=1800)
    hand_point = choose_point(fusion_state.filtered_hand_center_base, selected_hand.palm_center_base)
    draw_specs = [(display_grasp_point, (0, 255, 0), 'grasp'), (hand_point, (255, 120, 0), 'hand')]
    for point_base, color_bgr, _label in draw_specs:
        pixel = project_base_point_to_camera(point_base, frame_bundle.intrinsics, camera_transform, image_bgr.shape[1], image_bgr.shape[0])
        if pixel is None:
            continue
        cv.circle(image_bgr, pixel, 6, color_bgr, -1, cv.LINE_AA)
    return draw_sam3d_bbox_overlay(image_bgr, pipeline, 0 if camera_label == 'cam0' else 1)

def format_record_clock(elapsed_s):
    if elapsed_s is None:
        return None
    elapsed_s = max(float(elapsed_s), 0.0)
    minutes = int(elapsed_s // 60.0)
    seconds = int(elapsed_s % 60.0)
    tenths = int((elapsed_s - int(elapsed_s)) * 10.0)
    return f'REC {minutes:02d}:{seconds:02d}.{tenths:d}'

def draw_record_clock_overlay(image_bgr, elapsed_s):
    """Draw the task recording clock onto a preview frame."""
    clock_text = format_record_clock(elapsed_s)
    if not clock_text:
        return image_bgr
    origin = (18, 42)
    cv.putText(image_bgr, clock_text, origin, cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv.LINE_AA)
    cv.putText(image_bgr, clock_text, origin, cv.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 255), 2, cv.LINE_AA)
    return image_bgr

def draw_tactile_status_overlay(image_bgr, tactile_manager):
    if tactile_manager is None or not bool(getattr(tactile_manager, 'enabled', False)):
        return image_bgr
    tactile_snapshot = tactile_manager.snapshot(refresh=False) if hasattr(tactile_manager, 'snapshot') else None
    norm_value = getattr(tactile_manager, 'latest_norm', 0.0) if tactile_snapshot is None else tactile_snapshot.get('total_norm', 0.0)
    norm_text = f'{float(norm_value):.1f}'
    ref_norm = getattr(tactile_manager, 'release_reference_norm', None) if tactile_snapshot is None else tactile_snapshot.get('release_ref_norm')
    ref_text = '-' if ref_norm is None else f'{float(ref_norm):.1f}'
    delta_norm = getattr(tactile_manager, 'release_delta_norm', None) if tactile_snapshot is None else tactile_snapshot.get('release_delta_norm')
    delta_text = '-' if delta_norm is None else f'{float(delta_norm):.1f}'
    status = str(getattr(tactile_manager, 'release_status', 'off') if tactile_snapshot is None else tactile_snapshot.get('status', 'off'))
    text = f'Tactile norm={norm_text} ref={ref_text} d={delta_text} {status}'
    origin = (12, 74)
    cv.putText(image_bgr, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image_bgr, text, origin, cv.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv.LINE_AA)
    return image_bgr

def render_cam0_perception_debug(snapshot, pipeline, merged_object, selected_hand, fusion_state, display_object_point, display_grasp_point, shared_state, record_elapsed_s=None, tactile_manager=None):
    """Render the main cam0 preview with mask, object cloud, hand, and grasp."""
    image_bgr = np.asarray(snapshot.cam0.color_image).copy()
    object_debug = getattr(pipeline['object_worker_cam0'], 'last_debug', None)
    combined_mask = getattr(object_debug, 'combined_mask', None)
    if combined_mask is not None:
        mask_bool = np.asarray(combined_mask, dtype=bool)
        if mask_bool.shape[:2] == image_bgr.shape[:2]:
            overlay = np.zeros_like(image_bgr, dtype=np.uint8)
            overlay[mask_bool] = np.array([0, 180, 255], dtype=np.uint8)
            image_bgr = cv.addWeighted(image_bgr, 1.0, overlay, 0.35, 0.0)
    image_bgr = draw_projected_template_overlay(image_bgr, merged_object.merged_points_base, snapshot.cam0.intrinsics, pipeline['t_cam0_base'], max_points=1800)
    object_point = display_object_point
    hand_point = choose_point(fusion_state.filtered_hand_center_base, selected_hand.palm_center_base)
    draw_specs = [(object_point, (0, 140, 255), 'obj'), (display_grasp_point, (0, 255, 0), 'grasp'), (hand_point, (255, 120, 0), 'hand')]
    for point_base, color_bgr, label in draw_specs:
        pixel = project_base_point_to_camera(point_base, snapshot.cam0.intrinsics, pipeline['t_cam0_base'], image_bgr.shape[1], image_bgr.shape[0])
        if pixel is None:
            continue
        cv.circle(image_bgr, pixel, 6, color_bgr, -1, cv.LINE_AA)
        cv.putText(image_bgr, label, (pixel[0] + 8, pixel[1] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(image_bgr, label, (pixel[0] + 8, pixel[1] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.55, color_bgr, 1, cv.LINE_AA)
    image_bgr = draw_record_clock_overlay(image_bgr, record_elapsed_s)
    image_bgr = draw_tactile_status_overlay(image_bgr, tactile_manager)
    return draw_sam3d_bbox_overlay(image_bgr, pipeline, 0)

def main():
    """Run the full dual-camera handover loop and handle keyboard controls."""
    args = parse_args()
    config = load_yaml_config(args.config)
    args = apply_config_defaults(args, config)
    print(f"[INFO] Handover mode: {'SAM2/SAM3D'} (geometry={'fitted_template'}, grasp_detection={args.grasp_detection_mode}, release_parameter_mm={args.release_parameter_mm:.1f}, place_grasp_offset_xy={args.apply_place_grasp_offset_xy})")
    if args.place_object_xyz_mm is not None:
        print(f'[INFO] Place mode=fixed_xyz, TCP approach={args.place_object_xyz_mm} mm; descend={args.place_descend_mm:g} mm before open.')
    runtime_template_path = None
    pipeline = build_dual_perception_pipeline(args, runtime_template_path=runtime_template_path)
    sensor_hub = pipeline['sensor_hub']
    sam3d_live_runtime = None
    tactile_manager = AnySkinTactileManager(args) if args.enable_follow and bool(getattr(args, 'tactile_enabled', False)) else None
    try:
        readiness_strategy = READINESS_SILHOUETTE
        sam3d_live_runtime = Sam3DLiveRuntime(pipeline, args.config, readiness_strategy=readiness_strategy, regeneration_enabled=not False)
        sam3d_live_runtime.start()
        pipeline['sam3d_live_runtime'] = sam3d_live_runtime
        print(f'[INFO] SAM2 point tracking ready; template initialization strategy={readiness_strategy}.')
        if tactile_manager is not None:
            tactile_manager.start()
        sensor_hub.start()
    except BaseException:
        if tactile_manager is not None:
            tactile_manager.close()
        if sam3d_live_runtime is not None:
            sam3d_live_runtime.close()
        for hand_worker_key in ('hand_worker_cam0', 'hand_worker_cam1'):
            hand_worker = pipeline.get(hand_worker_key)
            if hand_worker is not None:
                try:
                    hand_worker.close()
                except Exception:
                    pass
        try:
            sensor_hub.stop()
        except Exception:
            pass
        raise
    tracking_model = args.model if not pipeline['prompt_classes'] else f"{args.model} ({','.join(pipeline['prompt_classes'])})"
    runtime_profiler = RuntimeProfiler(enabled=bool(args.profile_runtime), output_dir=args.profile_dir, sample_interval_s=args.profile_sample_interval_s, print_every_s=args.profile_print_every_s, run_context=build_runtime_profile_context(args, tracking_model, pipeline))
    if runtime_profiler.enabled:
        print(f"[INFO] Runtime profiler enabled. Press 's' to save logs to {args.profile_dir}.")
    cam0_window_name = 'cam0_grasp_target_follow'
    cam1_window_name = 'cam1_view'
    smoothed_fps = 0.0
    last_loop_time = time.perf_counter()
    shared_state = FollowSharedState(args)
    pipeline['shared_state'] = shared_state
    metadata_recorder = HandoverMetadataRecorder()
    video_recorder = None
    robot_worker = None
    robot_status = RobotStatus()
    last_task_ready_epoch = 0
    last_reset_done_epoch = 0
    last_task_done_epoch = 0
    pending_reset_reason = None
    pending_sam3d_regeneration = False
    pending_same_object_refit = False
    grasp_request_pending = False
    place_z_wait_signature = None
    active_task_epoch = None
    task_record_start_perf = None
    debug_3d_enabled = bool(args.debug_3d) and (not bool(args.disable_debug_3d_recording))
    if args.save_image and (not debug_3d_enabled):
        print('[WARN] --save-image requires --3d-debug; raw image/depth capture is disabled.')
    debug_3d_recorder = None
    if debug_3d_enabled:
        debug_3d_recorder = Debug3DRecorder(output_dir=args.debug_3d_dir, enabled=True, save_images=bool(args.save_image), max_object_points=args.debug_3d_max_object_points, max_template_points=args.debug_3d_max_template_points, record_template_axes=bool(args.debug_3d_template_axes))
        print(f"[INFO] 3D debug recorder armed. Press 'd' to save to {args.debug_3d_dir} and reset.")
        if debug_3d_recorder.save_images:
            print('[INFO] 3D debug recorder will include raw color/depth frames.')
    video_recorder_web_ui_enabled = get_video_recorder_web_ui_enabled(config)
    video_recorder_tactile_logging = get_video_recorder_tactile_logging_config(config)
    if 'baseline' == ABLATION_TACTILE_SENSING:
        video_recorder_tactile_logging = {'enabled': False, 'csv_enabled': False, 'rerun_enabled': False, 'rerun_live': False}
    if args.record_video:
        try:
            video_recorder = HandoverVideoRecorderService(serial=VIDEO_RECORDER_SERIAL, port=VIDEO_RECORDER_PORT, external_frames=True, metadata_recorder=metadata_recorder, web_ui_enabled=video_recorder_web_ui_enabled, tactile_manager=tactile_manager, tactile_logging_enabled=video_recorder_tactile_logging['enabled'], tactile_csv_enabled=video_recorder_tactile_logging['csv_enabled'], tactile_rerun_enabled=video_recorder_tactile_logging['rerun_enabled'], tactile_rerun_live=video_recorder_tactile_logging['rerun_live'])
            recorder_status = video_recorder.start_server()
            if video_recorder.is_web_ui_enabled:
                print(f'[INFO] Video recorder UI ready: {recorder_status}')
            else:
                print(f'[INFO] Video recorder ready; web UI disabled; press s to save directly. {recorder_status}')
        except Exception as exc:
            video_recorder = None
            print(f'[WARN] Failed to start video recorder service: {exc}')
    if args.enable_follow:
        robot_worker = RobotWorker(args, shared_state, metadata_recorder=metadata_recorder, tactile_manager=tactile_manager)
        robot_worker.start()
        robot_worker.submit(RobotRequest(ROBOT_REQ_INIT_ROBOT))
        robot_worker.submit(RobotRequest(ROBOT_REQ_START_FOLLOW))
    try:
        while True:
            current_time = time.time()
            loop_perf = time.perf_counter()
            loop_total_start = time.perf_counter()
            with shared_state.lock:
                current_task_epoch = int(shared_state.task_epoch)
                motion_triggered = bool(shared_state.motion_triggered)
            if robot_worker is not None:
                robot_status = robot_worker.get_status()
                if robot_status.reset_done_epoch != last_reset_done_epoch:
                    last_reset_done_epoch = int(robot_status.reset_done_epoch)
                    if pending_same_object_refit:
                        with shared_state.lock:
                            refit_task_id = int(shared_state.task_epoch)
                        if not begin_same_object_refit(pipeline, refit_task_id):
                            shared_state.clear_target(reset_prediction=True,reset_arm=True)
                            shared_state.stop_follow()
                            robot_worker.submit(RobotRequest(ROBOT_REQ_STOP_FOLLOW))
                            print('[WARN] Same-object refit could not start: no reusable template remains.')
                        pending_same_object_refit = False
                    else:
                        reset_perception_pipeline_for_system_reset(pipeline)
                    if debug_3d_recorder is not None:
                        debug_3d_recorder.clear()
                    if pending_reset_reason is not None:
                        discard_runtime_profile(runtime_profiler, reason=pending_reset_reason)
                        pending_reset_reason = None
                    grasp_request_pending = False
                    place_z_wait_signature = None
                    print('[INFO] Main-thread perception reset complete after robot reset.')
                    if pending_sam3d_regeneration and sam3d_live_runtime is not None:
                        shared_state.stop_follow()
                        robot_worker.submit(RobotRequest(ROBOT_REQ_STOP_FOLLOW))
                        with shared_state.lock:
                            regeneration_task_id = int(shared_state.task_epoch)
                        if sam3d_live_runtime.begin_regeneration(regeneration_task_id):
                            print('[INFO] SAM3D regeneration capturing started at HOME.')
                        pending_sam3d_regeneration = False
                if robot_status.task_ready_epoch != last_task_ready_epoch:
                    last_task_ready_epoch = int(robot_status.task_ready_epoch)
                    task_ready_timestamp = metadata_recorder.mark_task_ready(shared_state)
                    task_record_start_perf = time.perf_counter()
                    print(f'[INFO] Metadata task start timestamp={task_ready_timestamp}')
                    start_task_video_recording(video_recorder, task_ready_timestamp)
                if robot_status.task_done_epoch != last_task_done_epoch:
                    last_task_done_epoch = int(robot_status.task_done_epoch)
                    grasp_request_pending = False
                    place_z_wait_signature = None
                if robot_status.state == ROBOT_STATE_ERROR:
                    grasp_request_pending = False
            with shared_state.lock:
                current_task_epoch=int(shared_state.task_epoch)
            if sam3d_live_runtime is not None:
                capture_allowed=robot_worker is None or (robot_status.home_ready and robot_status.state in {ROBOT_STATE_IDLE,ROBOT_STATE_FOLLOWING})
                sam3d_live_runtime.set_capture_allowed(capture_allowed)
            runtime_profiler.start_frame(timestamp_unix_s=current_time, task_epoch=current_task_epoch)
            if current_task_epoch != active_task_epoch:
                if sam3d_live_runtime is not None:
                    sam3d_live_runtime.on_task_change(current_task_epoch,preserve_template=pending_same_object_refit)
                pipeline['hand_relative_fallback'].reset()
                pipeline['grasp_z_stabilizer'].reset()
                active_task_epoch = current_task_epoch
            if sam3d_live_runtime is not None:
                regeneration_event = sam3d_live_runtime.poll_regeneration()
                if regeneration_event == 'installed':
                    reset_perception_pipeline_for_system_reset(pipeline, preserve_tracking=True)
                    shared_state.clear_target(reset_prediction=True, reset_arm=True)
                    print('[INFO] New SAM3D tracker installed; waiting for SAM2 fitting and fresh SAM2 masks. Follow remains disabled.')
                elif regeneration_event == 'failed':
                    reset_perception_pipeline_for_system_reset(pipeline, preserve_tracking=True)
                    shared_state.clear_target(reset_prediction=True, reset_arm=True)
                    print(f'[WARN] SAM3D regeneration failed; robot target remains gated. Follow remains disabled. {sam3d_live_runtime.debug().last_error}')
            with runtime_profiler.stage('read_pair'):
                if sam3d_live_runtime is not None and sam3d_live_runtime.needs_fresh_pair():
                    settle_s=float(config.get('runtime',{}).get('prompt_capture_settle_sec',0.2))
                    snapshot=sensor_hub.read_fresh_pair(settle_s=settle_s)
                    print('[INFO] Fresh RGB/depth pair acquired for object selection after robot readiness.')
                else:
                    snapshot = sensor_hub.read_next_pair()
                if video_recorder is not None:
                    video_recorder.push_frame(snapshot.cam1.color_image)
            sam3d_gate_blocked = False
            if sam3d_live_runtime is not None:
                with runtime_profiler.stage('sam2_prompt'):
                    sam3d_gate_blocked = sam3d_live_runtime.before_frame(snapshot, task_id=current_task_epoch, now_ros_s=current_time)
            with runtime_profiler.stage('object_cam0'):
                object_cam0 = pipeline['object_worker_cam0'].process_frame(snapshot.cam0, frame_id=snapshot.pair_index)
            with runtime_profiler.stage('object_cam1'):
                object_cam1 = pipeline['object_worker_cam1'].process_frame(snapshot.cam1, frame_id=snapshot.pair_index)
            if sam3d_live_runtime is not None:
                sam3d_gate_blocked = bool(sam3d_live_runtime.after_object_perception(object_cam0, object_cam1) or sam3d_gate_blocked)
            with runtime_profiler.stage('hand_cam0'):
                hand_cam0 = pipeline['hand_worker_cam0'].process_frame(snapshot.cam0, frame_id=snapshot.pair_index)
            with runtime_profiler.stage('hand_cam1'):
                hand_cam1 = pipeline['hand_worker_cam1'].process_frame(snapshot.cam1, frame_id=snapshot.pair_index)
            with runtime_profiler.stage('merge'):
                merged_object = pipeline['object_merger'].process_states(object_cam0, object_cam1)
            with runtime_profiler.stage('shape_fit'):
                freeze_silhouette_scale = bool(sam3d_live_runtime is not None and sam3d_live_runtime.silhouette_scale_frozen)
                silhouette_observations = build_silhouette_observations(snapshot, pipeline)
                shape_fitting_state = pipeline['shape_fitting_tracker'].process(merged_object, silhouette_observations=silhouette_observations, freeze_silhouette_scale=freeze_silhouette_scale)
                initialization_ready = bool(sam3d_live_runtime is not None and sam3d_live_runtime.after_shape_fit(shape_fitting_state))
                if sam3d_live_runtime is not None:
                    sam3d_live_runtime.capture_regeneration_frame(snapshot)
                if initialization_ready:
                    print('[INFO] Initial geometry ready; switching from fixed prompts to persistent SAM2 tracking.')
                    sam3d_gate_blocked = True
                metadata_recorder.update_geometry(shape_fitting_state, now_perf=loop_perf)
            with runtime_profiler.stage('hand_select'):
                object_center_base = shape_fitting_state.centroid_base if bool(getattr(shape_fitting_state, 'valid', False)) else None
                selected_hand = pipeline['hand_selector'].process_states(hand_cam0, hand_cam1, object_center_base=object_center_base)
            object_debug_cam0 = getattr(pipeline['object_worker_cam0'], 'last_debug', None)
            with runtime_profiler.stage('fusion_grasp'):
                active_merged_object = build_fitted_merged_object(merged_object, shape_fitting_state)
                fusion_state = pipeline['fusion'].process_states(active_merged_object, selected_hand, now_timestamp=current_time)
                grasp_target = pipeline['grasp_planner'].process_states(active_merged_object, selected_hand, fusion_state)
                measured_object_point_base = choose_point(fusion_state.filtered_object_centroid_base, active_merged_object.centroid_base) if active_merged_object.valid else None
                measured_grasp_point_base = grasp_target.target_position_base if grasp_target.valid and active_merged_object.valid else None
                measured_grasp_point_base = offset_point_base_mm(measured_grasp_point_base, y_mm=GRASP_POINT_Y_OFFSET_MM, z_mm=GRASP_POINT_Z_OFFSET_MM)
            eef_pose_base = None
            eef_xyz_mm = None
            need_robot_pose = measured_grasp_point_base is not None or measured_object_point_base is not None or debug_3d_recorder is not None
            if robot_worker is not None and need_robot_pose:
                with runtime_profiler.stage('robot_read'):
                    robot_status = robot_worker.get_status()
                if robot_status.last_robot_pose is not None:
                    eef_pose_base = tuple((float(v) for v in robot_status.last_robot_pose))
                    eef_xyz_mm = meters_to_mm(eef_pose_base[:3])
            with runtime_profiler.stage('grasp_z_stabilization'):
                reference_xyz_mm = None if measured_grasp_point_base is None else meters_to_mm(measured_grasp_point_base)
                measured_grasp_point_base = pipeline['grasp_z_stabilizer'].process(measured_grasp_point_base, reference_xyz_mm=reference_xyz_mm, eef_xyz_mm=eef_xyz_mm)
            frame_record_elapsed_s = None if task_record_start_perf is None else max(float(loop_perf) - float(task_record_start_perf), 0.0)
            with runtime_profiler.stage('fallback'):
                hand_relative_fallback_state = pipeline['hand_relative_fallback'].process(measured_object_position_base=measured_object_point_base, measured_grasp_position_base=measured_grasp_point_base, selected_hand=selected_hand, fusion_state=fusion_state, motion_triggered=motion_triggered, now_timestamp=current_time, frame_id=snapshot.pair_index, record_elapsed_s=frame_record_elapsed_s)
            object_point_base = measured_object_point_base
            grasp_point_base = measured_grasp_point_base
            measurement_source = 'measured'
            if measured_object_point_base is None and hand_relative_fallback_state.valid:
                object_point_base = hand_relative_fallback_state.object_position_base
                grasp_point_base = hand_relative_fallback_state.grasp_position_base
                measurement_source = 'hand_fallback'
            object_pixel = project_base_point_to_camera(object_point_base, snapshot.cam0.intrinsics, pipeline['t_cam0_base'], snapshot.cam0.color_image.shape[1], snapshot.cam0.color_image.shape[0])
            if robot_worker is not None and eef_xyz_mm is None and (object_point_base is not None or debug_3d_recorder is not None):
                with runtime_profiler.stage('robot_read'):
                    robot_status = robot_worker.get_status()
                if robot_status.last_robot_pose is not None:
                    eef_pose_base = tuple((float(v) for v in robot_status.last_robot_pose))
                    eef_xyz_mm = meters_to_mm(eef_pose_base[:3])
            if debug_3d_recorder is not None:
                with runtime_profiler.stage('debug_3d'):
                    append_debug_3d_frame(debug_3d_recorder, snapshot=snapshot, current_time=current_time, loop_perf=loop_perf, record_elapsed_s=frame_record_elapsed_s, current_task_epoch=current_task_epoch, pipeline=pipeline, selected_hand=selected_hand, merged_object=merged_object, shape_fitting_state=shape_fitting_state, object_point_base=object_point_base, grasp_point_base=grasp_point_base, eef_pose_base=eef_pose_base, measurement_source=measurement_source, tactile_manager=tactile_manager)
            if sam3d_gate_blocked:
                object_point_base = None
                grasp_point_base = None
                measurement_source = 'sam3d_gate_blocked'
                enforce_sam3d_target_gate(shared_state, True)
            if object_point_base is not None:
                with runtime_profiler.stage('target_update'):
                    shared_state.update_place_z_samples(grasp_point_base=grasp_point_base, object_point_base=object_point_base, fitted_points_base=shape_fitting_state.fitted_points_base)
                    shared_state.update_target(grasp_point_base, object_point_base, object_pixel, eef_xyz_mm=eef_xyz_mm, measurement_source=measurement_source, object_label=merged_object.label)
                if robot_worker is not None and (not grasp_request_pending) and (robot_status.state == ROBOT_STATE_FOLLOWING) and shared_state.is_pregrasp_pose_reached(robot_status.last_robot_pose):
                    place_z_result = shared_state.finalize_place_z_from_recent_samples()
                    if not place_z_result['valid']:
                        wait_signature = (place_z_result['reason'], place_z_result['grasp_sample_count'], place_z_result['template_bottom_sample_count'])
                        if wait_signature != place_z_wait_signature:
                            print(f"[WARN] DIRECT GRASP waiting for place Z samples: reason={place_z_result['reason']} grasp_samples={place_z_result['grasp_sample_count']} template_bottom_samples={place_z_result['template_bottom_sample_count']}")
                            place_z_wait_signature = wait_signature
                    else:
                        place_z_wait_signature = None
                        if place_z_result['reason'] == 'fixed_xyz':
                            print(f'[INFO] Fixed box release ready: TCP XYZ={args.place_object_xyz_mm} mm')
                        else:
                            print(f"[INFO] frozen_place_z_mm prepared before grasp: grasp_z_med={place_z_result['grasp_z_median_mm']:.1f}, template_bottom_z_med={place_z_result['template_bottom_z_median_mm']:.1f}, release_parameter={place_z_result['release_parameter_mm']:.1f}, raw_place_z={place_z_result['raw_place_z_mm']:.1f}, place_z={place_z_result['place_z_mm']:.1f}")
                        print('[INFO] DIRECT GRASP trigger')
                        fitted_points_copy = None
                        if shape_fitting_state.fitted_points_base is not None:
                            fitted_points_copy = np.asarray(shape_fitting_state.fitted_points_base, dtype=np.float32).copy()
                        grasp_point_copy = None if grasp_point_base is None else tuple((float(v) for v in grasp_point_base))
                        template_axes = getattr(shape_fitting_state, 'template_axes_base', None)
                        template_axes_copy = None if template_axes is None else np.asarray(template_axes, dtype=np.float32).copy()
                        action_context = RobotActionContext(fitted_points_base=fitted_points_copy, grasp_point_base=grasp_point_copy, object_label=shape_fitting_state.label, template_axes_base=template_axes_copy)
                        robot_worker.submit(RobotRequest(ROBOT_REQ_START_GRASP_PLACE, payload={'context': action_context}))
                        grasp_request_pending = True
            else:
                shared_state.clear_target(reset_prediction=bool(sam3d_gate_blocked), reset_arm=bool(sam3d_gate_blocked))
            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_loop_time, 1e-06)
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            last_loop_time = now
            record_elapsed_s = None if task_record_start_perf is None else max(now - task_record_start_perf, 0.0)
            with runtime_profiler.stage('render'):
                if tactile_manager is not None:
                    tactile_manager.total_norm()
                annotated = render_cam0_perception_debug(snapshot, pipeline, active_merged_object, selected_hand, fusion_state, object_point_base, grasp_point_base, shared_state, record_elapsed_s=record_elapsed_s, tactile_manager=tactile_manager)
                cam1_preview = render_camera_mask_preview(snapshot, pipeline, pipeline['object_worker_cam1'], active_merged_object, selected_hand, fusion_state, grasp_point_base, camera_label='cam1')
                cv.imshow(cam0_window_name, annotated)
                cv.imshow(cam1_window_name, cam1_preview)
            with runtime_profiler.stage('wait_key'):
                key = cv.waitKey(1) & 255
            runtime_profiler.add_stage_ms('loop_total', (time.perf_counter() - loop_total_start) * 1000.0)
            runtime_profiler.record_frame(frame_index=int(snapshot.pair_index), timestamp_unix_s=current_time, task_epoch=current_task_epoch, fps=smoothed_fps, metrics=collect_runtime_profile_metrics(snapshot=snapshot, pipeline=pipeline, object_cam0=object_cam0, object_cam1=object_cam1, merged_object=merged_object, shape_fitting_state=shape_fitting_state, selected_hand=selected_hand, fusion_state=fusion_state, grasp_target=grasp_target, measurement_source=measurement_source, instant_fps=instant_fps, smoothed_fps=smoothed_fps))
            if key in (27, ord('q')):
                break
            if key == ord('f'):
                if robot_worker is not None:
                    follow_enabled = bool(shared_state.get_snapshot()['follow_enabled'])
                    request_type = ROBOT_REQ_STOP_FOLLOW if follow_enabled else ROBOT_REQ_START_FOLLOW
                    robot_worker.submit(RobotRequest(request_type))
                else:
                    shared_state.clear_follow_pause()
                    shared_state.toggle_follow()
            elif key == ord('g'):
                if sam3d_live_runtime is None:
                    print('[WARN] SAM3D regeneration is only available with --object-backend sam3d.')
                elif sam3d_live_runtime.regeneration_state in {'capturing', 'generating', 'fitting'}:
                    print(f'[WARN] SAM3D regeneration is already running: state={sam3d_live_runtime.regeneration_state}')
                else:
                    pending_same_object_refit = False
                    print('[INFO] SAM3D regeneration requested; stopping follow and returning HOME.')
                    shared_state.clear_target(reset_prediction=True, reset_arm=True)
                    shared_state.stop_follow()
                    if robot_worker is not None:
                        pending_sam3d_regeneration = True
                        robot_worker.submit(RobotRequest(ROBOT_REQ_RESET_HOME))
                    else:
                        shared_state.reset_for_restart(follow_enabled=False)
                        reset_perception_pipeline_for_system_reset(pipeline)
                        with shared_state.lock:
                            regeneration_task_id = int(shared_state.task_epoch)
                        if sam3d_live_runtime.begin_regeneration(regeneration_task_id):
                            print('[INFO] SAM3D regeneration capturing started (perception-only mode).')
            elif key == ord('p'):
                shared_state.clear_target(reset_prediction=True, reset_arm=True)
                shared_state.stop_follow()
                if robot_worker is not None:
                    robot_worker.submit(RobotRequest(ROBOT_REQ_STOP_FOLLOW))
                if sam3d_live_runtime.regeneration_state == 'refitting':
                    sam3d_live_runtime.regeneration_state = 'fitting'
                sam3d_live_runtime.selection_requested = True
            elif key == ord('h'):
                if not sam3d_live_runtime.can_refit_existing_template:
                    print('[WARN] H needs an existing SAM3D template. Finish the first template generation before reusing it.')
                    continue
                if pending_same_object_refit:
                    print('[INFO] Same-object refit is already waiting for HOME.')
                    continue
                if pending_reset_reason is not None or pending_sam3d_regeneration or (
                    robot_worker is not None and robot_worker.get_status().state in {
                        ROBOT_STATE_INITIALIZING,ROBOT_STATE_RESETTING,ROBOT_STATE_GRASPING,
                        ROBOT_STATE_RETURNING,ROBOT_STATE_PLACING,ROBOT_STATE_STOPPING,
                    }
                ):
                    print('[WARN] Same-object refit waits for the current robot/reset action to finish; press H again afterward.')
                    continue
                print('[INFO] Same-object handover requested: returning HOME and reusing the SAM3D template and SAM2 tracking.')
                pending_sam3d_regeneration = False
                shared_state.clear_target(reset_prediction=True,reset_arm=True)
                shared_state.stop_follow()
                discard_video_recording_for_reset(video_recorder)
                task_record_start_perf = None
                if robot_worker is not None:
                    pending_same_object_refit = True
                    pending_reset_reason = 'h_key_same_object_refit'
                    robot_worker.submit(RobotRequest(ROBOT_REQ_RESET_HOME))
                else:
                    reset_tactile_state_for_system_reset(tactile_manager)
                    shared_state.reset_for_restart(follow_enabled=False)
                    with shared_state.lock:
                        refit_task_id = int(shared_state.task_epoch)
                    begin_same_object_refit(pipeline, refit_task_id)
                    if debug_3d_recorder is not None:
                        debug_3d_recorder.clear()
                    discard_runtime_profile(runtime_profiler, reason='h_key_same_object_refit')
            elif key == ord('r'):
                pending_same_object_refit = False
                print('[INFO] Reset requested: returning to startup state')
                discard_video_recording_for_reset(video_recorder)
                task_record_start_perf = None
                if robot_worker is not None:
                    pending_reset_reason = 'r_key_reset'
                    robot_worker.submit(RobotRequest(ROBOT_REQ_RESET_HOME))
                else:
                    reset_tactile_state_for_system_reset(tactile_manager)
                    shared_state.reset_for_restart(follow_enabled=False)
                    reset_perception_pipeline_for_system_reset(pipeline)
                    if debug_3d_recorder is not None:
                        debug_3d_recorder.clear()
                    discard_runtime_profile(runtime_profiler, reason='r_key_reset')
            elif key == ord('d'):
                if debug_3d_recorder is None:
                    print('[WARN] 3D debug recording is disabled; launch with --3d-debug.')
                    continue
                debug_save_path = None
                try:
                    debug_save_path = debug_3d_recorder.save()
                    print(f'[INFO] 3D debug recording saved: {debug_save_path}')
                except Exception as exc:
                    print(f'[WARN] 3D debug save failed; keeping buffered frames and skipping reset: {exc}')
                    continue
                debug_3d_recorder.clear()
                pending_same_object_refit = False
                discard_video_recording_for_reset(video_recorder)
                task_record_start_perf = None
                if robot_worker is not None:
                    pending_reset_reason = 'd_key_debug_reset'
                    robot_worker.submit(RobotRequest(ROBOT_REQ_RESET_HOME))
                else:
                    reset_tactile_state_for_system_reset(tactile_manager)
                    shared_state.reset_for_restart(follow_enabled=False)
                    reset_perception_pipeline_for_system_reset(pipeline)
                    discard_runtime_profile(runtime_profiler, reason='d_key_debug_reset')
            elif key == ord('s'):
                if robot_worker is not None:
                    robot_worker.submit(RobotRequest(ROBOT_REQ_SAVE_AND_STOP))
                else:
                    shared_state.request_follow_pause()
                    shared_state.stop_follow()
                if metadata_recorder is not None:
                    csv_path = metadata_recorder.record_completion()
                    if csv_path is not None:
                        print(f'[INFO] Handover metadata appended: {csv_path}')
                    else:
                        print('[INFO] Handover metadata was already saved or no task metadata is available.')
                if video_recorder is not None:
                    if video_recorder.is_web_ui_enabled:
                        ok, message = video_recorder.finish_recording_to_pending()
                    else:
                        ok, message = video_recorder.finish_and_save_plain_recording()
                    level = '[INFO]' if ok else '[WARN]'
                    print(f'{level} Video recorder: {message}')
                    if video_recorder.is_web_ui_enabled:
                        open_video_recorder_ui(video_recorder)
                save_runtime_profile(runtime_profiler, reason='s_key')
    finally:
        runtime_profiler.close()
        if debug_3d_recorder is not None:
            debug_3d_recorder.close()
        shared_state.stop_event.set()
        if robot_worker is not None:
            robot_worker.submit(RobotRequest(ROBOT_REQ_SHUTDOWN))
            robot_worker.join(timeout=args.move_timeout_s + 2.0)
        if tactile_manager is not None:
            tactile_manager.close()
        if sam3d_live_runtime is not None:
            sam3d_live_runtime.close()
        try:
            pipeline['hand_worker_cam0'].close()
        finally:
            try:
                pipeline['hand_worker_cam1'].close()
            finally:
                sensor_hub.stop()
        if video_recorder is not None:
            video_recorder.stop()
        cv.destroyAllWindows()

def configure_teaching(args, config):
    global HOME_JOINTS_DEG, EEF_X_OFFSET_MM, EEF_Y_OFFSET_MM, EEF_Z_OFFSET_MM, BACKOFF_X_MM
    if not args.enable_follow:
        return
    from robot.xarm_controller import load_teach
    teach = load_teach(config, args.config)
    HOME_JOINTS_DEG = np.rad2deg(teach['home_joints_rad']).tolist()
    EEF_X_OFFSET_MM, EEF_Y_OFFSET_MM, EEF_Z_OFFSET_MM = teach['approach_offset_mm']
    BACKOFF_X_MM = teach['backoff_x_mm']
    config['robot']['return_sequence']['place_object_xy_mm'] = teach['place_object_xy_mm']
    config['robot']['return_sequence']['place_object_xyz_mm'] = teach.get('place_object_xyz_mm')
    config['robot']['return_sequence']['place_descend_mm'] = teach.get('place_descend_mm', 0.0)
    config['safety']['workspace_bounds_m'] = teach['workspace_bounds_m']

def draw_sam3d_bbox_overlay(image_bgr, pipeline, camera_id):
    runtime = pipeline.get('sam3d_live_runtime')
    if runtime is None:
        return image_bgr
    debug = runtime.debug()
    text = f'SAM2 | template {debug.initialization_phase} {debug.scaling_valid_frames}/{debug.scaling_required_frames} | P: prompt'
    cv.putText(image_bgr, text, (10, image_bgr.shape[0] - 16), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv.LINE_AA)
    return image_bgr

def build_runtime_profile_context(args, tracking_model, pipeline):
    return {'config_path': str(Path(args.config).resolve()), 'tracking_model': tracking_model, 'object_backend': 'sam3d', 'enable_follow': args.enable_follow, 'control_hz': args.control_hz, 'coordinate_frame': 'xarm_base', 'orientation_format': 'rpy_rad'}
if __name__ == '__main__':
    main()
