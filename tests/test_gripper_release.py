"""Release status/communication regressions, without hardware or real sleeps."""
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

import handover
from robot.xarm_controller import (
    GripperActivationLostError,
    GripperCommunicationError,
    XArmController,
)
from tests.test_handover_cycle import arguments
from tests.test_robotiq_communication import clocked
from tests.test_xarm_controller import config, sdk


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.arm = sdk()
        self.controller = XArmController(config(), sdk=self.arm)
        self.controller.connect()
        self.addCleanup(self.controller.close)

    def reports(self, *reports):
        remaining = iter(reports)
        def read():
            self.arm.robotiq_status = dict(next(remaining))
            return 0, [9, 3, 6, 0, 0, 0, 0, 0, 0]
        self.arm.robotiq_get_status.side_effect = read

    @staticmethod
    def status(**changes):
        return {'gACT':1, 'gSTA':3, 'gFLT':0, 'gPR':0, 'gGTO':1, 'gPO':0, 'gOBJ':3, **changes}

    def test_prepare_settle_and_slow_open_confirm_before_contact_metadata(self):
        self.controller.config['robot']['gripper']['open_prepare_delay_sec'] = .5
        self.reports(self.status(gPO=150, gPR=255, gOBJ=2),
                     self.status(gPO=120, gOBJ=0), self.status(gPO=40, gOBJ=0), self.status())
        metadata = Mock()
        read_times = []
        with clocked() as clock, redirect_stdout(StringIO()):
            start = clock.now
            read = self.arm.robotiq_get_status.side_effect
            def timed_read():
                read_times.append(clock.now)
                metadata.note_robot_last_contact.assert_not_called()
                return read()
            self.arm.robotiq_get_status.side_effect = timed_read
            self.assertTrue(handover.execute_gripper_open(self.controller, dwell_s=0, metadata_recorder=metadata))
        self.assertGreaterEqual(read_times[0] - start, .5)
        self.assertGreaterEqual(read_times[1] - read_times[0], .2 - 1e-9)
        metadata.note_robot_last_contact.assert_called_once()
        self.arm.robotiq_set_position.assert_called_once()

    def test_default_open_does_not_add_an_idle_delay(self):
        self.arm.robotiq_status.update(self.status())
        read_times = []
        with clocked() as clock, redirect_stdout(StringIO()):
            start = clock.now
            self.arm.robotiq_get_status.side_effect = lambda: (read_times.append(clock.now) or 0)
            self.assertTrue(self.controller.open_gripper_blocking())
        self.assertEqual(read_times[0], start)
        self.assertGreaterEqual(read_times[1] - read_times[0], .2 - 1e-9)

    def test_contact_stopped_at_98_does_not_reissue_a_position_move(self):
        for obj in (1, 2, 3):
            with self.subTest(gOBJ=obj), clocked(), redirect_stdout(StringIO()):
                self.arm.robotiq_status.update(self.status(gPO=98, gPR=255, gOBJ=obj))
                self.assertTrue(self.controller.stop_gripper_motion())
                self.arm.robotiq_set_position.assert_not_called()

    def test_moving_gripper_still_receives_stop_at_current_position(self):
        self.arm.robotiq_status.update(self.status(gPO=98, gPR=255, gOBJ=0))
        with clocked():
            self.assertTrue(self.controller.stop_gripper_motion())
        self.arm.robotiq_set_position.assert_called_once()
        self.assertEqual(self.arm.robotiq_set_position.call_args.args, (98,))
        self.assertEqual(self.controller._gripper_io_history[-1]['request']['position'], 98)

    def test_closed_request_echo_cannot_count_as_open_even_at_zero(self):
        self.arm.robotiq_status.update(self.status(gPR=255))
        with clocked(), redirect_stdout(StringIO()), self.assertRaisesRegex(TimeoutError, 'open timeout'):
            self.controller.open_gripper_blocking()
        self.arm.robotiq_set_position.assert_called_once()

    def test_open_obstruction_aborts_without_another_go_to_or_contact_metadata(self):
        self.reports(self.status(gPR=255, gPO=150, gOBJ=2), self.status(gPO=130, gOBJ=1))
        metadata = Mock()
        with clocked(), redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, 'blocked by contact'):
            handover.execute_gripper_open(self.controller, metadata_recorder=metadata)
        self.arm.robotiq_set_position.assert_called_once()
        self.arm.set_state.assert_called_with(4)
        metadata.note_robot_last_contact.assert_not_called()

    def test_slow_open_timeout_does_not_replace_open_with_partly_closed_target(self):
        self.arm.robotiq_status.update(self.status(gPO=40, gOBJ=0))
        with clocked(), redirect_stdout(StringIO()), self.assertRaises(TimeoutError):
            self.controller.open_gripper_blocking()
        self.arm.robotiq_set_position.assert_called_once()
        self.assertEqual(self.arm.robotiq_set_position.call_args.args[0], 0)

    def test_c19_before_open_reports_stage_and_never_sends_open(self):
        def failed_read():
            self.arm.get_err_warn_code.return_value = 0, [19, 0]
            return 23, []
        self.arm.robotiq_get_status.side_effect = failed_read
        metadata = Mock()
        output = StringIO()
        with clocked(), redirect_stdout(output), self.assertRaises(GripperCommunicationError):
            handover.execute_gripper_open(self.controller, metadata_recorder=metadata)
        self.assertIn('phase=open_prepare', output.getvalue())
        self.assertIn('response_length=0', output.getvalue())
        self.assertIn('port_readback=', output.getvalue())
        self.assertIn('recent transactions', output.getvalue())
        self.arm.robotiq_set_position.assert_not_called()
        self.arm.clean_error.assert_not_called()
        metadata.note_robot_last_contact.assert_not_called()

    def test_c19_during_open_wait_never_retries_or_sends_hold(self):
        self.arm.robotiq_status.update(self.status(gPO=150, gPR=255, gOBJ=2))
        calls = 0
        def read():
            nonlocal calls
            calls += 1
            if calls == 1:
                return 0
            self.arm.get_err_warn_code.return_value = 0, [19, 0]
            return 23, []
        self.arm.robotiq_get_status.side_effect = read
        with clocked(), redirect_stdout(StringIO()), self.assertRaisesRegex(GripperCommunicationError, 'phase=open_wait'):
            handover.execute_gripper_open(self.controller)
        self.arm.robotiq_set_position.assert_called_once()
        self.arm.robotiq_get_status.assert_called_with()
        self.assertEqual(calls, 2)
        self.arm.clean_error.assert_not_called()
        with self.assertRaises(GripperCommunicationError):
            self.controller.open_gripper_blocking()
        self.assertEqual(calls, 2)

    def test_transaction_history_is_bounded(self):
        with clocked():
            for _ in range(50):
                self.controller.get_gripper_close_state()
        self.assertEqual(len(self.controller._gripper_io_history), 20)

    def test_place_failure_never_retracts_or_returns_home(self):
        args = arguments()
        args.pre_release_descend_before_open = False
        shared = handover.FollowSharedState(args)
        shared.fixed_orientation_base = (0, 0, 0)
        shared.initial_pose_base = (.4, 0, .3, 0, 0, 0)
        shared.frozen_place_z_mm = 160.
        self.arm.robotiq_get_status.return_value = 23, []
        metadata = Mock()
        with clocked(), redirect_stdout(StringIO()), \
                patch.object(handover, 'move_robot_and_wait', return_value=True) as move, \
                patch.object(handover, 'move_robot_to_home_pose') as home, \
                self.assertRaises(GripperCommunicationError):
            handover.execute_return_and_place(self.controller, shared, args, metadata_recorder=metadata)
        self.assertEqual([c.kwargs['source_mode'] for c in move.call_args_list], ['return_hover', 'return_place'])
        home.assert_not_called()
        metadata.note_robot_last_contact.assert_not_called()
        self.arm.robotiq_set_position.assert_not_called()

    def test_zero_status_after_startup_cannot_reset_reactivate_or_open(self):
        self.arm.robotiq_status.update(self.status())
        with clocked(), redirect_stdout(StringIO()):
            self.assertTrue(self.controller.open_gripper_blocking())
            self.arm.reset_mock()
            self.arm.robotiq_status = dict.fromkeys(self.status(), 0)
            metadata = Mock()
            with self.assertRaisesRegex(GripperActivationLostError, 'phase=open_prepare'):
                handover.execute_gripper_open(self.controller, metadata_recorder=metadata)
            reads = self.arm.robotiq_get_status.call_count
            for operation in (self.controller.open_gripper_blocking,
                              self.controller.start_gripper_close,
                              self.controller.stop_gripper_motion,
                              self.controller.ensure_gripper_ready):
                with self.assertRaises(GripperActivationLostError):
                    operation()
            self.assertFalse(self.controller.read_robot_state().valid)
        self.assertEqual(reads, 1)
        self.assertEqual(self.arm.robotiq_get_status.call_count, reads)
        self.arm.robotiq_reset.assert_not_called()
        self.arm.robotiq_set_activate.assert_not_called()
        self.arm.robotiq_set_position.assert_not_called()
        metadata.note_robot_last_contact.assert_not_called()
        self.arm.set_state.assert_called_with(4)

    def test_deactivation_during_runtime_open_wait_cannot_retry_activation(self):
        with clocked(), redirect_stdout(StringIO()):
            self.controller.start_gripper_close()
            self.arm.reset_mock()
            self.reports(self.status(gPO=122, gPR=122), self.status(gACT=0, gSTA=0, gFLT=5))
            with self.assertRaisesRegex(GripperActivationLostError, 'phase=open_wait'):
                handover.execute_gripper_open(self.controller)
        self.arm.robotiq_set_position.assert_called_once()
        self.assertEqual(self.arm.robotiq_set_position.call_args.args, (0,))
        self.arm.robotiq_reset.assert_not_called()
        self.arm.robotiq_set_activate.assert_not_called()

    def test_runtime_idle_fault_nine_with_activation_retained_can_open(self):
        with clocked(), redirect_stdout(StringIO()):
            self.controller.start_gripper_close()
            self.arm.reset_mock()
            self.reports(self.status(gPO=122, gPR=122, gFLT=9), self.status())
            self.assertTrue(self.controller.open_gripper_blocking())
        self.arm.robotiq_set_position.assert_called_once()
        self.arm.robotiq_reset.assert_not_called()
        self.arm.robotiq_set_activate.assert_not_called()

    def test_runtime_fault_five_with_active_bits_cannot_reset(self):
        with clocked(), redirect_stdout(StringIO()):
            self.controller.start_gripper_close()
            self.arm.reset_mock()
            self.arm.robotiq_status.update(self.status(gFLT=5))
            with self.assertRaises(GripperActivationLostError):
                self.controller.ensure_gripper_ready()
        self.arm.robotiq_reset.assert_not_called()
        self.arm.robotiq_set_activate.assert_not_called()
        self.arm.robotiq_set_position.assert_not_called()

    def test_deactivation_during_close_is_not_a_successful_grasp(self):
        self.arm.robotiq_status = dict.fromkeys(self.status(), 0)
        with clocked(), redirect_stdout(StringIO()):
            self.assertFalse(handover.execute_gripper_close(self.controller, verbose=False))
        self.arm.robotiq_set_position.assert_called_once()
        self.arm.robotiq_reset.assert_not_called()
        self.arm.robotiq_set_activate.assert_not_called()
        self.arm.set_state.assert_called_with(4)
