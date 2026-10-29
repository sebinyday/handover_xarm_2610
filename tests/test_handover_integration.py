"""Exercise the ported orchestration without opening cameras or robot sockets."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import tempfile
import time
import unittest
import numpy as np
import yaml
import handover
from pipeline import PendingTemplateTracker
from perception.object_merger import ObjectMerger
from perception.hand_selector import HandSelector
from perception.fusion import PerceptionFusion
from perception.grasp_target import GraspTargetPlanner
from perception.grasp_z_stabilizer import GraspPointZStabilizer
from perception.hand_relative_fallback import HandRelativeFallbackTracker
from system.shared_state import ObjectState,HandState
from utils.realsense_stream import FrameBundle
from video_record import HandoverVideoRecorderService
from tests.test_xarm_controller import sdk,config
from robot.xarm_controller import XArmController


class OrchestrationTests(unittest.TestCase):
    def test_inactive_gripper_recovers_before_home_and_follow(self):
        with patch('sys.argv',['handover.py','--enable-follow']):
            args=handover.parse_args()
        args=handover.apply_config_defaults(args,config())
        shared=handover.FollowSharedState(args)
        worker=handover.RobotWorker(args,shared)
        arm=sdk()
        controller=XArmController(config(),sdk=arm)
        inactive=dict(gPO=3,gOBJ=0,gFLT=5,gACT=0,gSTA=0,gPR=0,gGTO=1)
        ready=dict(inactive,gFLT=0,gACT=1,gSTA=3)
        reports=iter([inactive,inactive,dict(inactive,gACT=1,gSTA=1),ready,dict(ready,gOBJ=3)])
        def read_status():
            arm.robotiq_status=next(reports)
            return 0
        def move_home(**kwargs):
            arm.get_servo_angle.return_value=(0,kwargs['angle'])
            return 0
        arm.robotiq_get_status.side_effect=read_status
        arm.set_servo_angle.side_effect=move_home
        with patch.object(handover.XArmController,'from_config',return_value=controller),patch('robot.xarm_controller.time.sleep'):
            worker._handle_request(handover.RobotRequest(handover.ROBOT_REQ_INIT_ROBOT))
            worker._handle_request(handover.RobotRequest(handover.ROBOT_REQ_START_FOLLOW))
        self.assertEqual(worker.get_status().state,handover.ROBOT_STATE_FOLLOWING)
        self.assertEqual(worker.get_status().task_ready_epoch,1)
        self.assertTrue(shared.get_snapshot()['follow_enabled'])
        calls=[call[0] for call in arm.mock_calls]
        self.assertLess(calls.index('robotiq_set_position'),calls.index('set_servo_angle'))
        arm.set_servo_angle.assert_called_once()
        self.assertEqual(len(arm.set_servo_angle.call_args.kwargs['angle']),7)

    def test_startup_gripper_failure_disconnects_and_never_issues_home(self):
        with patch('sys.argv',['handover.py','--enable-follow']):
            args=handover.parse_args()
        args=handover.apply_config_defaults(args,config())
        shared=handover.FollowSharedState(args)
        worker=handover.RobotWorker(args,shared)
        controller=XArmController(config(),sdk=sdk())
        with patch.object(handover.XArmController,'from_config',return_value=controller),patch.object(controller,'open_gripper_blocking',side_effect=TimeoutError('Robotiq activation timeout')) as open_gripper:
            worker._handle_request(handover.RobotRequest(handover.ROBOT_REQ_INIT_ROBOT))
            worker._handle_request(handover.RobotRequest(handover.ROBOT_REQ_START_FOLLOW))
        open_gripper.assert_called_once()
        controller.arm.disconnect.assert_called_once()
        controller.arm.set_servo_angle.assert_not_called()
        self.assertEqual(worker.get_status().state,handover.ROBOT_STATE_ERROR)
        self.assertIn('activation timeout',worker.get_status().last_error)
        self.assertFalse(shared.get_snapshot()['follow_enabled'])

    def test_failed_home_cannot_start_queued_follow(self):
        with patch('sys.argv',['handover.py','--enable-follow']):
            args=handover.parse_args()
        args=handover.apply_config_defaults(args,config())
        shared=handover.FollowSharedState(args)
        worker=handover.RobotWorker(args,shared)
        controller=XArmController(config(),sdk=sdk())
        controller.connect()
        with patch.object(handover,'init_xarm',return_value=controller),patch.object(handover,'move_robot_to_home_pose',side_effect=RuntimeError('HOME failed')),patch.object(handover,'disconnect_xarm') as disconnect:
            worker._handle_request(handover.RobotRequest(handover.ROBOT_REQ_INIT_ROBOT))
            worker._handle_request(handover.RobotRequest(handover.ROBOT_REQ_START_FOLLOW))
        disconnect.assert_called_once_with(controller)
        self.assertIsNone(worker.controller)
        self.assertEqual(worker.get_status().state,handover.ROBOT_STATE_ERROR)
        self.assertIn('HOME failed',worker.get_status().last_error)
        self.assertFalse(shared.get_snapshot()['follow_enabled'])
        self.assertEqual(worker.get_status().task_ready_epoch,0)

    def test_perception_loop_and_cleanup(self):
        config=yaml.safe_load(Path('configs/handover.yaml').read_text())
        frame=FrameBundle(np.zeros((32,32,3),np.uint8),np.ones((32,32),np.float32),dict(fx=100,fy=100,cx=16,cy=16),time.time()*1000,'test')
        snapshot=SimpleNamespace(cam0=frame,cam1=frame,pair_index=1,timestamp_delta_ms=0,within_sync_tolerance=True)
        hub=Mock();hub.read_next_pair.return_value=snapshot
        pipeline={'sensor_hub':hub,'shape_fitting_tracker':PendingTemplateTracker(),'prompt_classes':['object'],'object_backend':'sam3d','ablation_mode':'baseline','runtime_template_path':None,'t_cam0_base':np.eye(4),'t_cam1_base':np.eye(4),'transform_chain':SimpleNamespace(t_base_cam0=np.eye(4),t_base_cam1=np.eye(4))}
        for i in range(2):
            worker=Mock();worker.process_frame.return_value=ObjectState(camera_id=i);worker.last_debug=None
            pipeline[f'object_worker_cam{i}']=worker
            hand=Mock();hand.process_frame.return_value=HandState(camera_id=i)
            pipeline[f'hand_worker_cam{i}']=hand
        for name,cls in [('hand_selector',HandSelector),('object_merger',ObjectMerger),('fusion',PerceptionFusion),('grasp_planner',GraspTargetPlanner),('grasp_z_stabilizer',GraspPointZStabilizer),('hand_relative_fallback',HandRelativeFallbackTracker)]:pipeline[name]=cls(config)
        runtime=Mock();runtime.poll_regeneration.return_value=None;runtime.needs_fresh_pair.return_value=False;runtime.before_frame.return_value=True;runtime.after_object_perception.return_value=True;runtime.after_shape_fit.return_value=False;runtime.silhouette_scale_frozen=False
        runtime.debug.return_value=SimpleNamespace(initialization_phase='capturing',scaling_valid_frames=0,scaling_required_frames=4)
        with patch('sys.argv',['handover.py','--profile-runtime']),patch.object(handover,'build_dual_perception_pipeline',return_value=pipeline),patch.object(handover,'Sam3DLiveRuntime',return_value=runtime),patch.object(handover.cv,'imshow'),patch.object(handover.cv,'destroyAllWindows'),patch.object(handover.cv,'waitKey',side_effect=[-1,ord('q')]):
            handover.main()
        self.assertEqual(hub.read_next_pair.call_count,2)
        hub.stop.assert_called_once();runtime.close.assert_called_once()
        for i in range(2):pipeline[f'hand_worker_cam{i}'].close.assert_called_once()

    def test_cancelled_move_stops_before_sdk_motion(self):
        from threading import Event
        arm=sdk();controller=XArmController(config(),sdk=arm);controller.connect()
        cancel=Event();cancel.set()
        ok=handover.move_robot_and_wait(controller,(.3,0,.4),(0,0,0),timeout_s=1,tolerance_m=.01,source_mode='test',cancel_event=cancel)
        self.assertFalse(ok);arm.set_position.assert_not_called();arm.set_state.assert_called_with(4)

    def test_home_wait_accepts_seven_joints(self):
        arm=sdk();controller=XArmController(config(),sdk=arm);controller.connect()
        self.assertTrue(handover.wait_until_joint_target_reached(controller,[0]*7,timeout_s=.1,tolerance_rad=.001))

    def test_move_timeout_stops(self):
        arm=sdk();controller=XArmController(config(),sdk=arm);controller.connect()
        ok=handover.move_robot_and_wait(controller,(.5,0,.4),(0,0,0),timeout_s=.01,tolerance_m=.001,source_mode='test')
        self.assertFalse(ok);arm.set_state.assert_called_with(4)

    def test_recorder_accepts_hub_frames_without_reopening_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            service=HandoverVideoRecorderService(save_dir=directory,external_frames=True,web_ui_enabled=False,tactile_logging_enabled=False)
            frame=np.zeros((480,640,3),np.uint8)
            service.push_frame(frame)
            with patch.object(service,'_start_camera') as start:
                service.start_capture()
                start.assert_not_called()
            self.assertIsNone(service._pipeline)
            self.assertIsNotNone(service._latest_frame)
            service.stop()

    def test_metadata_rpy_and_no_vision_fill_fields(self):
        from scipy.spatial.transform import Rotation
        from utils.handover_metadata import rpy_to_quaternion_xyzw,METADATA_COLUMNS
        np.testing.assert_allclose(rpy_to_quaternion_xyzw([.2,.4,.6]),Rotation.from_euler('xyz',[.2,.4,.6]).as_quat())
        self.assertFalse(any('fill_level' in name or 'mass_full_est' in name for name in METADATA_COLUMNS))

    def test_worker_keeps_processing_after_sdk_failure(self):
        import sys
        with patch.object(sys,'argv',['handover.py']):args=handover.parse_args()
        args=handover.apply_config_defaults(args,config())
        shared=handover.FollowSharedState(args)
        worker=handover.RobotWorker(args,shared)
        with patch.object(worker,'_handle_start_follow',side_effect=RuntimeError('SDK failed')),patch.object(worker,'_safe_stop') as stop:
            worker._handle_request(handover.RobotRequest(handover.ROBOT_REQ_START_FOLLOW))
            self.assertEqual(worker.get_status().state,handover.ROBOT_STATE_ERROR)
            stop.assert_called_once()
        worker._handle_request(handover.RobotRequest(handover.ROBOT_REQ_SHUTDOWN))
        self.assertTrue(worker._shutdown_event.is_set())
