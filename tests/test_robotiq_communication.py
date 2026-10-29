"""Tool communication and failed-grasp regressions; no hardware connection."""

import unittest
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

import handover
from robot.xarm_controller import GripperCommunicationError, XArmController
from tests.test_xarm_controller import config, sdk


class Clock:
    def __init__(self):
        self.now = 1000.

    def sleep(self, seconds):
        assert seconds >= 0
        self.now += seconds


@contextmanager
def clocked():
    clock = Clock()
    with patch('time.monotonic', side_effect=lambda: clock.now), patch('time.time', side_effect=lambda: clock.now), patch('time.sleep', side_effect=clock.sleep):
        yield clock


class RobotiqCommunicationTests(unittest.TestCase):
    def setUp(self):
        self.arm = sdk()
        self.controller = XArmController(config(), sdk=self.arm)
        self.addCleanup(self.controller.close)

    def statuses(self, *reports):
        remaining = iter(reports)
        def read():
            self.arm.robotiq_status = dict(next(remaining))
            return 0
        self.arm.robotiq_get_status.side_effect = read

    def test_matching_baud_is_verified_without_reboot_or_reset(self):
        self.controller.connect()
        self.arm.set_checkset_default_baud.assert_called_once_with(3, 115200)
        self.arm.set_rs485_baudrate.assert_not_called()
        self.arm.set_rs485_timeout.assert_called_once_with(500, target='robot', protocol='modbus_rtu')
        self.arm.robotiq_reset.assert_not_called()
        self.arm.clean_error.assert_not_called()
        names = [call[0] for call in self.arm.mock_calls]
        self.assertLess(names.index('get_rs485_timeout'), names.index('motion_enable'))

    def test_wrong_baud_is_configured_once_before_motion_enable_and_read_back(self):
        self.arm.get_rs485_baudrate.side_effect = [(0, 2000000), (0, 115200)]
        with clocked() as clock:
            self.controller.connect()
            self.assertGreaterEqual(clock.now, 1001.)
        self.arm.set_rs485_baudrate.assert_called_once_with(115200, target='robot')
        names = [call[0] for call in self.arm.mock_calls]
        self.assertLess(names.index('set_rs485_baudrate'), names.index('motion_enable'))
        self.arm.robotiq_set_position.assert_not_called()

    def test_failed_baud_readback_never_enables_motion(self):
        self.arm.get_rs485_baudrate.return_value = (0, 9600)
        with clocked(), self.assertRaisesRegex(RuntimeError, 'baudrate mismatch'):
            self.controller.connect()
        self.arm.motion_enable.assert_not_called()
        self.arm.robotiq_set_position.assert_not_called()

    def test_failed_timeout_readback_never_enables_motion(self):
        self.arm.get_rs485_timeout.return_value = (0, 20)
        with self.assertRaisesRegex(RuntimeError, 'timeout mismatch'):
            self.controller.connect()
        self.arm.motion_enable.assert_not_called()

    def test_existing_c19_is_not_automatically_cleared_or_reconfigured(self):
        self.arm.get_err_warn_code.return_value = (0, [19, 0])
        with self.assertRaisesRegex(GripperCommunicationError, 'C19 is already active'):
            self.controller.connect()
        self.arm.clean_error.assert_not_called()
        self.arm.set_rs485_baudrate.assert_not_called()
        self.arm.motion_enable.assert_not_called()

    def test_transactions_observe_command_settle_and_minimum_status_spacing(self):
        self.controller.connect()
        seen = []
        with clocked() as clock:
            self.arm.robotiq_set_position.side_effect = lambda *a, **kw: (seen.append(('write', clock.now)) or 0)
            self.arm.robotiq_get_status.side_effect = lambda: (seen.append(('read', clock.now)) or 0)
            self.controller.start_gripper_close()
            self.controller.get_gripper_close_state()
            self.controller.get_gripper_close_state()
        self.assertGreaterEqual(seen[1][1] - seen[0][1], .2 - 1e-9)
        self.assertGreaterEqual(seen[2][1] - seen[1][1], .1 - 1e-9)

    def test_reset_and_activation_have_settle_time_before_status_poll(self):
        self.controller.connect()
        inactive = {'gPO':0, 'gOBJ':0, 'gFLT':5, 'gACT':0, 'gSTA':0}
        ready = dict(inactive, gFLT=0, gACT=1, gSTA=3)
        self.statuses(inactive, inactive, ready)
        seen = []
        with clocked() as clock:
            read = self.arm.robotiq_get_status.side_effect
            self.arm.robotiq_get_status.side_effect = lambda: (seen.append(('read', clock.now)) or read())
            self.arm.robotiq_reset.side_effect = lambda: (seen.append(('reset', clock.now)) or 0)
            self.arm.robotiq_set_activate.side_effect = lambda **kw: (seen.append(('activate', clock.now)) or 0)
            self.controller.ensure_gripper_ready()
        times = {name:stamp for name, stamp in seen if name != 'read'}
        reads = [stamp for name, stamp in seen if name == 'read']
        self.assertGreaterEqual(reads[1] - times['reset'], .5 - 1e-9)
        self.assertGreaterEqual(reads[2] - times['activate'], 2.0 - 1e-9)

    def test_failed_write_latches_operation_and_c19_without_retrying_gripper(self):
        self.controller.connect()
        def fail(*args, **kwargs):
            self.arm.get_err_warn_code.return_value = (0, [19, 0])
            return 23, []
        self.arm.robotiq_set_position.side_effect = fail
        with clocked(), self.assertRaisesRegex(GripperCommunicationError, 'robotiq_set_position.*C19'):
            self.controller.start_gripper_close()
        first_error = self.controller.gripper_communication_error
        self.assertIn('code=23', str(first_error))
        for operation in (self.controller.start_gripper_close, self.controller.get_gripper_close_state, self.controller.stop_gripper_motion):
            with self.assertRaises(GripperCommunicationError):
                operation()
        self.controller.read_robot_state()
        self.assertIs(self.controller.gripper_communication_error, first_error)
        self.arm.robotiq_set_position.assert_called_once()
        self.arm.robotiq_get_status.assert_not_called()
        self.arm.robotiq_reset.assert_not_called()
        self.arm.clean_error.assert_not_called()

    def test_status_timeout_latches_and_prevents_followup_hold_write(self):
        self.controller.connect()
        self.arm.robotiq_get_status.return_value = (-1, [])
        with clocked(), self.assertRaises(GripperCommunicationError):
            self.controller.get_gripper_close_state()
        with self.assertRaises(GripperCommunicationError):
            self.controller.stop_gripper_motion()
        self.arm.robotiq_get_status.assert_called_once()
        self.arm.robotiq_set_position.assert_not_called()

    def test_close_failure_does_not_fall_back_to_another_close_command(self):
        self.controller.connect()
        self.arm.robotiq_set_position.return_value = 23, []
        with clocked(), redirect_stdout(StringIO()):
            self.assertFalse(handover.execute_gripper_close(self.controller, verbose=False))
        self.arm.robotiq_set_position.assert_called_once()
        self.arm.robotiq_get_status.assert_not_called()
        self.arm.set_state.assert_called_with(4)

    def test_failed_status_aborts_close_without_read_storm(self):
        self.controller.connect()
        self.arm.robotiq_get_status.return_value = -1, []
        with clocked(), redirect_stdout(StringIO()):
            self.assertFalse(handover.execute_gripper_close(self.controller, verbose=False))
        self.arm.robotiq_get_status.assert_called_once()
        self.arm.robotiq_set_position.assert_called_once()
        self.arm.set_state.assert_called_with(4)

    def test_empty_close_is_failure_even_after_encoder_threshold_and_stall(self):
        self.controller.connect()
        moving = {'gPO':225, 'gOBJ':0, 'gFLT':0, 'gACT':1, 'gSTA':3}
        empty = dict(moving, gPO=255, gOBJ=3)
        self.statuses(moving, moving, moving, empty)
        output = StringIO()
        with clocked(), redirect_stdout(output):
            result = handover.execute_gripper_close(self.controller, timeout_s=5, verbose=False)
        self.assertFalse(result)
        self.assertIn('without contact evidence', output.getvalue())
        self.arm.robotiq_set_position.assert_called_once()
        self.assertEqual(self.arm.robotiq_get_status.call_count, 4)
        self.arm.clean_error.assert_not_called()

    def test_inner_contact_still_verifies_a_grasp(self):
        self.controller.connect()
        self.arm.robotiq_status.update(gPO=140, gOBJ=2)
        with clocked(), redirect_stdout(StringIO()):
            self.assertTrue(handover.execute_gripper_close(self.controller, timeout_s=5, verbose=False))
        self.arm.robotiq_set_position.assert_called_once()
        self.assertEqual(self.arm.robotiq_set_position.call_args.args, (255,))

    def test_anyskin_contact_is_preserved_when_robotiq_reports_terminal_position(self):
        self.controller.connect()
        self.arm.robotiq_status.update(gPO=255, gOBJ=3)
        tactile = SimpleNamespace(enabled=True, total_norm=lambda: 100., release_status='')
        with clocked(), redirect_stdout(StringIO()), patch.object(handover, 'print_tactile_frame_log'):
            self.assertTrue(handover.execute_gripper_close(self.controller, timeout_s=5, verbose=False, tactile_manager=tactile, tactile_contact_threshold=35.))
        self.arm.robotiq_set_position.assert_called_once()

    def test_terminal_stop_does_not_send_another_position_request(self):
        self.controller.connect()
        self.arm.robotiq_status.update(gPO=255, gOBJ=3)
        self.assertTrue(self.controller.stop_gripper_motion())
        self.arm.robotiq_set_position.assert_not_called()

    def test_motion_timeout_is_separate_from_modbus_packet_timeout_and_cli_wins(self):
        with patch('sys.argv', ['handover.py']):
            args = handover.apply_config_defaults(handover.parse_args(), config())
        self.assertEqual(args.gripper_close_timeout_s, 5.)
        with patch('sys.argv', ['handover.py', '--gripper-close-timeout-s', '7']):
            args = handover.apply_config_defaults(handover.parse_args(), config())
        self.assertEqual(args.gripper_close_timeout_s, 7.)
        self.assertEqual(self.controller.gripper_modbus_timeout_ms, 500)


if __name__ == '__main__':
    unittest.main()
