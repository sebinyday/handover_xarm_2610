import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import yaml
from robot.xarm_controller import XArmController,load_teach
from system.shared_state import RobotCommandState,ROBOT_CMD_SERVO_TO_POSITION,ROBOT_CMD_MOVE_TO_POSITION


def config():
    return yaml.safe_load(Path('configs/handover.yaml').read_text())


def sdk():
    arm=Mock()
    arm.connected=True
    arm.tcp_offset=[0,0,174,0,0,0];arm.world_offset=[0]*6
    arm.get_err_warn_code.return_value=(0,[0,0])
    arm.get_state.return_value=(0,0)
    arm.get_position.return_value=(0,[300,-20,400,0.2,-0.3,0.4])
    arm.get_servo_angle.return_value=(0,[0]*7)
    arm.get_rs485_baudrate.return_value=(0,115200)
    arm.get_rs485_timeout.return_value=(0,500)
    arm.get_is_moving.return_value=False
    arm.robotiq_status={'gPO':100,'gOBJ':2,'gFLT':0,'gACT':1,'gSTA':3,'gPR':0,'gGTO':1}
    def register_report(callback):
        callback({})
        callback({})
        return True
    arm.register_report_callback.side_effect=register_report
    for name in ['set_mode','set_state','motion_enable','set_servo_angle','set_position','set_servo_cartesian','robotiq_set_position','robotiq_get_status','robotiq_reset','robotiq_set_activate','set_checkset_default_baud','set_rs485_baudrate','set_rs485_timeout']:
        getattr(arm,name).return_value=0
    return arm


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.arm=sdk();self.controller=XArmController(config(),sdk=self.arm);self.controller.connect()

    def command(self,kind=ROBOT_CMD_SERVO_TO_POSITION):
        return RobotCommandState(command_type=kind,target_position_base=(0.3,-0.02,0.4),fixed_orientation_base=(0.2,-0.3,0.4),valid=True)

    def test_meter_radian_boundary_and_seven_joints(self):
        state=self.controller.step(self.command())
        np.testing.assert_allclose(self.arm.set_servo_cartesian.call_args.args[0],[300,-20,400,.2,-.3,.4])
        np.testing.assert_allclose(state.actual_tcp_pose_base,[.3,-.02,.4,.2,-.3,.4])
        self.assertEqual(len(state.joint_positions),7)

    def test_servo_to_position_stops_then_changes_mode(self):
        self.controller.step(self.command())
        self.arm.reset_mock()
        self.controller.step(self.command(ROBOT_CMD_MOVE_TO_POSITION))
        calls=[(c[0],c[1]) for c in self.arm.mock_calls]
        stop=next(i for i,c in enumerate(calls) if c==('set_state',(4,)))
        mode=next(i for i,c in enumerate(calls) if c==('set_mode',(0,)))
        move=next(i for i,c in enumerate(calls) if c[0]=='set_position')
        self.assertLess(stop,mode);self.assertLess(mode,move)

    def test_home_is_seven_axis(self):
        with self.assertRaises(ValueError):self.controller.move_to_joint_positions([0]*6,speed_rad_s=.1,acceleration_rad_s2=.1)
        self.controller.move_to_joint_positions([0]*7,speed_rad_s=.1,acceleration_rad_s2=.1)
        self.assertEqual(len(self.arm.set_servo_angle.call_args.kwargs['angle']),7)

    def test_stale_command_stops(self):
        command=self.command();command.timestamp=time.time()-2
        with self.assertRaises(RuntimeError):self.controller.step(command)
        self.arm.set_servo_cartesian.assert_not_called()
        self.arm.set_state.assert_called_with(4)

    def test_workspace_rejected(self):
        command=self.command();command.target_position_base=(5,0,0)
        with self.assertRaises(ValueError):self.controller.step(command)
        self.arm.set_servo_cartesian.assert_not_called()

    def test_sdk_failure_propagates(self):
        self.arm.set_servo_cartesian.return_value=23
        with self.assertRaisesRegex(RuntimeError,'23'):self.controller.step(self.command())

    def test_robotiq_uses_parsed_status(self):
        status=self.controller.get_gripper_close_state()
        self.assertEqual(status['object_status'],'STOPPED_INNER_OBJECT')
        self.assertEqual(status['position'],100)
        self.arm.robotiq_status['gFLT']=9
        with self.assertRaises(RuntimeError):self.controller.get_gripper_close_state()

    def gripper_reports(self,*reports):
        pending=iter(reports)
        def read():
            self.arm.robotiq_status=dict(gPR=0,gGTO=1,**next(pending))
            return 0
        self.arm.robotiq_get_status.side_effect=read

    def test_inactive_gripper_is_activated_before_open_command(self):
        inactive=dict(gPO=3,gOBJ=0,gFLT=5,gACT=0,gSTA=0)
        pending=dict(inactive,gACT=1,gSTA=1)
        ready=dict(inactive,gFLT=0,gACT=1,gSTA=3)
        opened=dict(ready,gOBJ=3)
        self.gripper_reports(inactive,inactive,pending,ready,opened)
        with patch('robot.xarm_controller.time.sleep'):
            self.assertTrue(self.controller.open_gripper_blocking())
        calls=[call[0] for call in self.arm.mock_calls]
        self.assertLess(calls.index('robotiq_reset'),calls.index('robotiq_set_activate'))
        self.assertLess(calls.index('robotiq_set_activate'),calls.index('robotiq_set_position'))
        self.assertEqual(calls[:calls.index('robotiq_set_position')].count('robotiq_get_status'),4)
        self.arm.robotiq_set_activate.assert_called_once_with(wait=False)

    def test_already_active_gripper_is_preserved(self):
        self.arm.robotiq_status.update(gPO=3,gOBJ=3)
        self.assertTrue(self.controller.open_gripper_blocking())
        self.arm.robotiq_reset.assert_not_called()
        self.arm.robotiq_set_activate.assert_not_called()

    def test_pending_activation_is_waited_for_without_reset(self):
        pending=dict(gPO=3,gOBJ=0,gFLT=5,gACT=1,gSTA=1)
        ready=dict(pending,gFLT=0,gSTA=3)
        self.gripper_reports(pending,pending,ready)
        with patch('robot.xarm_controller.time.sleep'):
            self.controller.ensure_gripper_ready()
        self.arm.robotiq_reset.assert_not_called()
        self.arm.robotiq_set_activate.assert_not_called()

    def test_disabled_activation_rejects_inactive_gripper_before_open(self):
        self.controller.config['robot']['gripper']['activate_on_connect']=False
        self.arm.robotiq_status.update(gFLT=5,gACT=0,gSTA=0)
        with self.assertRaisesRegex(RuntimeError,'activate_on_connect'):
            self.controller.open_gripper_blocking()
        self.arm.robotiq_reset.assert_not_called()
        self.arm.robotiq_set_position.assert_not_called()

    def test_activation_timeout_cannot_send_open(self):
        self.arm.robotiq_status.update(gFLT=5,gACT=1,gSTA=1)
        self.controller.config['robot']['gripper']['activation_timeout_sec']=0
        with self.assertRaisesRegex(TimeoutError,'activation timeout'):
            self.controller.open_gripper_blocking()
        self.arm.robotiq_set_position.assert_not_called()

    def test_idle_communication_fault_and_activation_delay_do_not_finish_open(self):
        idle=dict(gPO=3,gOBJ=0,gFLT=9,gACT=1,gSTA=3)
        pending=dict(idle,gFLT=5,gSTA=1)
        ready=dict(idle,gFLT=0)
        self.gripper_reports(idle,pending,pending,ready,dict(ready,gOBJ=3))
        with patch('robot.xarm_controller.time.sleep'):
            self.assertTrue(self.controller.open_gripper_blocking())
        self.arm.robotiq_reset.assert_not_called()
        self.assertEqual(self.arm.robotiq_get_status.call_count,5)
        self.assertEqual(self.arm.robotiq_set_position.call_count,2)

    def test_hard_gripper_fault_never_sends_activation_or_open(self):
        self.arm.robotiq_status.update(gFLT=13)
        with self.assertRaisesRegex(RuntimeError,'Robotiq fault 13'):
            self.controller.open_gripper_blocking()
        self.arm.robotiq_reset.assert_not_called()
        self.arm.robotiq_set_position.assert_not_called()

    def test_stop_then_restart(self):
        self.controller.stop_joint_motion()
        self.arm.get_state.return_value=(0,4)
        self.controller.step(self.command())
        self.arm.set_mode.assert_called_with(1)

    def test_watchdog_stops_without_new_commands(self):
        self.controller.step(self.command())
        self.controller._servo_deadline=time.monotonic()-1
        self.controller.read_robot_state()
        self.arm.set_state.assert_called_with(4)
        self.assertTrue(self.controller.last_debug.watchdog_triggered)

    def test_tcp_mismatch_never_enables_motion(self):
        self.arm=sdk();self.arm.tcp_offset=[0]*6
        controller=XArmController(config(),sdk=self.arm)
        with self.assertRaises(RuntimeError):controller.connect()
        self.arm.motion_enable.assert_not_called()

    def test_connect_waits_for_reported_tcp_and_world_offsets(self):
        arm=sdk()
        arm.tcp_offset=[0]*6
        arm.world_offset=[0]*6
        settings=config()
        settings['robot']['xarm']['expected_world_offset_mm_rad']=[0,20,0,0,0,0]
        def register_report(callback):
            # The SDK starts with zeros; TCP is updated before the first
            # callback and world offset after that callback in a rich report.
            arm.tcp_offset=[0,0,174,0,0,0]
            callback({})
            arm.motion_enable.assert_not_called()
            arm.world_offset=[0,20,0,0,0,0]
            callback({})
            return True
        arm.register_report_callback.side_effect=register_report
        XArmController(settings,sdk=arm).connect()
        arm.motion_enable.assert_called_once_with(enable=True)
        arm.release_report_callback.assert_called_once()

    def test_missing_reports_never_enable_motion_even_if_cached_offsets_match(self):
        arm=sdk()
        arm.register_report_callback.side_effect=None
        arm.register_report_callback.return_value=True
        settings=config()
        settings['robot']['xarm']['report_timeout_sec']=0
        with self.assertRaisesRegex(TimeoutError,'status reports not received'):
            XArmController(settings,sdk=arm).connect()
        arm.motion_enable.assert_not_called()
        arm.set_mode.assert_not_called()
        arm.disconnect.assert_called_once()
        arm.release_report_callback.assert_called_once()

    def test_world_mismatch_reports_values_and_never_enables_motion(self):
        arm=sdk()
        arm.world_offset=[0,20,0,0,0,0]
        with self.assertRaisesRegex(RuntimeError,'world_offset.*actual=.*expected='):
            XArmController(config(),sdk=arm).connect()
        arm.motion_enable.assert_not_called()

    def test_missing_teaching_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            teach_path = Path(directory) / 'teach.yaml'
            teach_path.write_text('home_joints_rad: null\n')
            settings = config()
            settings['robot']['teach_file'] = str(teach_path)
            with self.assertRaisesRegex(ValueError, 'Teach home_joints'):
                load_teach(settings, 'configs/handover.yaml')

    def test_steady_reads_boolean_sdk_result(self):
        self.arm.get_is_moving.return_value=True
        self.assertFalse(self.controller.isSteady())
        self.arm.get_is_moving.return_value=False
        self.assertTrue(self.controller.isSteady())
