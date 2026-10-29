"""Regression checks for HOME capture, grasp approach and release sequencing."""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import handover
from object_pt_extraction.dual_realsense_manager import DualRealSenseManager
from perception.sam3d_live_runtime import Sam3DLiveRuntime
from pipeline import PendingTemplateTracker
from robot.xarm_controller import XArmController
from system.dual_sensor_hub import DualSensorHub
from system.shared_state import HandState, ObjectState
from tests.test_xarm_controller import config, sdk
from utils.realsense_stream import FrameBundle, RealSenseCamera


def arguments():
    with patch("sys.argv", ["handover.py", "--enable-follow"]):
        args = handover.apply_config_defaults(handover.parse_args(), config())
    # These fixtures exercise the retained XY + fitted-height release mode.
    args.place_object_xyz_mm = None
    return args


class CaptureTests(unittest.TestCase):
    def test_main_uses_new_pair_only_after_home_readiness(self):
        frame = FrameBundle(
            np.zeros((32, 32, 3), np.uint8),
            np.ones((32, 32), np.float32),
            {"fx": 100, "fy": 100, "cx": 16, "cy": 16},
            time.time() * 1000,
            "test",
        )
        old = SimpleNamespace(
            cam0=frame,
            cam1=frame,
            pair_index=1,
            timestamp_delta_ms=0,
            within_sync_tolerance=True,
        )
        fresh = SimpleNamespace(
            cam0=frame,
            cam1=frame,
            pair_index=2,
            timestamp_delta_ms=0,
            within_sync_tolerance=True,
        )
        hub = Mock()
        hub.read_next_pair.return_value = old
        hub.read_fresh_pair.return_value = fresh
        pipeline = {
            "sensor_hub": hub,
            "shape_fitting_tracker": PendingTemplateTracker(),
            "prompt_classes": ["object"],
            "object_backend": "sam3d",
            "ablation_mode": "baseline",
            "runtime_template_path": None,
            "t_cam0_base": np.eye(4),
            "t_cam1_base": np.eye(4),
        }
        for i in range(2):
            worker = Mock()
            worker.process_frame.return_value = ObjectState(camera_id=i)
            worker.last_debug = None
            pipeline[f"object_worker_cam{i}"] = worker
            hand = Mock()
            hand.process_frame.return_value = HandState(camera_id=i)
            pipeline[f"hand_worker_cam{i}"] = hand
        for key, cls in [
            ("hand_selector", handover.HandSelector),
            ("object_merger", handover.ObjectMerger),
            ("fusion", handover.PerceptionFusion),
            ("grasp_planner", handover.GraspTargetPlanner),
            ("grasp_z_stabilizer", handover.GraspPointZStabilizer),
            ("hand_relative_fallback", handover.HandRelativeFallbackTracker),
        ]:
            pipeline[key] = cls(config())
        runtime = Mock()
        runtime.poll_regeneration.return_value = None
        runtime.before_frame.return_value = True
        runtime.after_object_perception.return_value = True
        runtime.after_shape_fit.return_value = False
        runtime.silhouette_scale_frozen = False
        runtime.capture_allowed = False
        runtime.set_capture_allowed.side_effect = lambda value: setattr(
            runtime, "capture_allowed", value
        )
        runtime.needs_fresh_pair.side_effect = lambda: runtime.capture_allowed
        robot = Mock()
        robot.get_status.side_effect = [
            handover.RobotStatus(state=handover.ROBOT_STATE_INITIALIZING),
            handover.RobotStatus(state=handover.ROBOT_STATE_FOLLOWING, home_ready=True),
        ]
        with (
            patch("sys.argv", ["handover.py", "--enable-follow"]),
            patch.object(
                handover, "build_dual_perception_pipeline", return_value=pipeline
            ),
            patch.object(handover, "Sam3DLiveRuntime", return_value=runtime),
            patch.object(handover, "RobotWorker", return_value=robot),
            patch.object(handover, "AnySkinTactileManager"),
            patch.object(handover.cv, "imshow"),
            patch.object(handover.cv, "destroyAllWindows"),
            patch.object(handover.cv, "waitKey", side_effect=[-1, ord("q")]),
        ):
            handover.main()
        hub.read_next_pair.assert_called_once()
        hub.read_fresh_pair.assert_called_once_with(settle_s=0.2)
        self.assertEqual(
            [call.args[0].pair_index for call in runtime.before_frame.call_args_list],
            [1, 2],
        )
        self.assertEqual(
            [call.args[0] for call in runtime.set_capture_allowed.call_args_list],
            [False, True],
        )

    def test_home_joint_tolerance_does_not_publish_while_still_moving(self):
        arm = sdk()
        controller = XArmController(config(), sdk=arm)
        controller.connect()
        arm.get_is_moving.side_effect = [True, False]
        with patch.object(handover.time, "sleep"):
            self.assertTrue(
                handover.wait_until_joint_target_reached(
                    controller, [0] * 7, timeout_s=1, tolerance_rad=0.01
                )
            )
        self.assertEqual(arm.get_is_moving.call_count, 2)

    def test_reset_blocks_capture_before_worker_starts_the_queued_request(self):
        args = arguments()
        worker = handover.RobotWorker(args, handover.FollowSharedState(args))
        worker._publish_task_ready()
        self.assertTrue(worker.get_status().home_ready)
        worker.submit(handover.RobotRequest(handover.ROBOT_REQ_RESET_HOME))
        self.assertFalse(worker.get_status().home_ready)

    def test_selection_and_template_capture_are_deferred_until_home(self):
        workers = [Mock(), Mock()]
        for worker in workers:
            worker.tracker.initialized = False
        pipeline = {
            "object_worker_cam0": workers[0],
            "object_worker_cam1": workers[1],
            "transform_chain": Mock(),
        }
        runtime = Sam3DLiveRuntime(pipeline, "configs/handover.yaml")
        self.addCleanup(runtime.close)
        runtime.on_task_change(1)
        runtime.set_capture_allowed(False)
        old = SimpleNamespace(pair_index=10, cam0=Mock(), cam1=Mock())
        fresh = SimpleNamespace(pair_index=11, cam0=Mock(), cam1=Mock())
        with patch("perception.sam3d_live_runtime.select_points") as select:
            self.assertTrue(runtime.before_frame(old, task_id=1))
            runtime.capture_regeneration_frame(old)
            select.assert_not_called()
            self.assertIsNone(runtime.future)
            self.assertIsNone(runtime.last_pair)
            runtime.set_capture_allowed(True)
            self.assertTrue(runtime.needs_fresh_pair())
            runtime.before_frame(fresh, task_id=1)
            self.assertEqual(select.call_count, 2)
            self.assertTrue(all(call.args[2] == 11 for call in select.call_args_list))

    def test_fresh_pair_discards_both_camera_queues_and_cached_follower(self):
        manager = DualRealSenseManager.__new__(DualRealSenseManager)
        manager.cameras = {"a": Mock(), "b": Mock()}
        manager._paired_latest_frame = "old follower"
        hub = DualSensorHub.__new__(DualSensorHub)
        hub._manager = manager
        fresh = object()
        hub.read_next_pair = Mock(return_value=fresh)
        self.assertIs(hub.read_fresh_pair(settle_s=0), fresh)
        self.assertIsNone(manager._paired_latest_frame)
        for camera in manager.cameras.values():
            self.assertEqual(camera.discard_pending_frames.call_count, 2)
        hub.read_next_pair.assert_called_once()

    def test_realsense_drain_does_not_return_a_queued_frame(self):
        camera = RealSenseCamera.__new__(RealSenseCamera)
        camera.pipeline = Mock()
        camera.pipeline.poll_for_frames.side_effect = [True, True, False]
        camera.discard_pending_frames()
        self.assertEqual(camera.pipeline.poll_for_frames.call_count, 3)


class ApproachTests(unittest.TestCase):
    def setUp(self):
        self.shared = handover.FollowSharedState(arguments())
        self.shared.args.pregrasp_tolerance_mm = (10.0, 10.0, 10.0)
        self.shared.latest_target_xyz_mm = np.array([600.0, 0.0, 150.0])
        self.shared.latest_grasp_xyz_mm = np.array([600.0, 0.0, 150.0])
        self.shared.latest_target_t = time.perf_counter()
        self.shared.motion_triggered = True
        self.shared.valid_detection_streak = 5

    def test_does_not_close_with_legacy_190_mm_x_gap(self):
        self.assertFalse(self.shared.is_pregrasp_pose_reached((0.42, 0, 0.15, 0, 0, 0)))
        self.assertTrue(self.shared.is_pregrasp_pose_reached((0.595, 0, 0.15, 0, 0, 0)))

    def test_grasp_requires_motion_trigger_and_fresh_target(self):
        self.shared.motion_triggered = False
        self.assertFalse(self.shared.is_pregrasp_pose_reached((0.6, 0, 0.15, 0, 0, 0)))
        self.shared.motion_triggered = True
        self.shared.latest_target_t = time.perf_counter() - 2
        self.assertFalse(self.shared.is_pregrasp_pose_reached((0.6, 0, 0.15, 0, 0, 0)))

    def test_closure_compares_tcp_target_including_offsets(self):
        self.shared.latest_target_xyz_mm = np.array([620.0, 0.0, 150.0])
        self.assertFalse(self.shared.is_pregrasp_pose_reached((0.6, 0, 0.15, 0, 0, 0)))
        self.assertTrue(self.shared.is_pregrasp_pose_reached((0.62, 0, 0.15, 0, 0, 0)))


class PlaceTests(unittest.TestCase):
    def setUp(self):
        self.args = arguments()
        self.args.tactile_release_ref_delay_s = 0
        self.args.workspace_x = [300.0, 900.0]
        self.args.workspace_y = [-500.0, 500.0]
        self.args.workspace_z = [80.0, 800.0]
        self.args.tactile_release_descent_min_z_mm = 40.0
        self.settings = config()
        self.settings["safety"]["workspace_bounds_m"] = {
            "x": [0.3, 0.9],
            "y": [-0.5, 0.5],
            "z": [0.08, 0.8],
        }
        self.arm = sdk()
        self.arm.robotiq_status.update(gPO=0, gOBJ=3)
        self.controller = XArmController(self.settings, sdk=self.arm)
        self.controller.connect()
        self.tactile = SimpleNamespace(
            enabled=True,
            release_reference_norm=100.0,
            release_status="armed",
            total_norm=lambda: 100.0,
            update_release_delta=lambda value: 0.0,
            set_release_reference=lambda value: None,
        )

    def test_release_uses_workspace_floor_then_opens_and_returns_home(self):
        self.assertEqual(self.args.workspace_z[0], 80.0)
        self.assertEqual(self.args.tactile_release_descent_min_z_mm, 40.0)

        def move(*pose, **kwargs):
            self.arm.get_position.return_value = (0, list(pose))
            return 0

        self.arm.set_position.side_effect = move

        def home(**kwargs):
            self.arm.get_servo_angle.return_value = (0, kwargs["angle"])
            return 0

        self.arm.set_servo_angle.side_effect = home
        shared = handover.FollowSharedState(self.args)
        shared.fixed_orientation_base = (0, 0, 0)
        shared.initial_pose_base = (0.4, 0, 0.3, 0, 0, 0)
        shared.frozen_place_z_mm = 160.0
        with patch.object(handover, "print_tactile_frame_log"):
            self.assertTrue(
                handover.execute_return_and_place(
                    self.controller, shared, self.args, tactile_manager=self.tactile
                )
            )
        z_commands = [call.args[2] for call in self.arm.set_position.call_args_list]
        self.assertIn(80.0, z_commands)
        self.assertGreaterEqual(min(z_commands), 80.0)
        self.arm.robotiq_set_position.assert_called_once()
        self.assertEqual(self.arm.robotiq_set_position.call_args.args[0], 0)
        self.arm.set_servo_angle.assert_called_once()
        self.assertEqual(shared.get_snapshot()["task_state"], "DONE")

    def test_release_invalid_feedback_does_not_open_gripper(self):
        self.arm.get_position.return_value = (0, [600, 0, 100, 0, 0, 0])
        real_read = self.controller.read_robot_state

        def invalid(**kwargs):
            state = real_read(**kwargs)
            state.valid = False
            return state

        with patch.object(self.controller, "read_robot_state", side_effect=invalid):
            result = handover.execute_tactile_release_descent(
                self.controller, self.tactile, (0, 0, 0), (600, 0, 100), self.args
            )
        self.assertTrue(result["failed"])
        self.arm.robotiq_set_position.assert_not_called()
