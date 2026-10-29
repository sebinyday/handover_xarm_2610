"""Fixed XYZ box release, with no hardware connections."""
import csv
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import yaml

import handover
from robot.xarm_controller import XArmController, load_teach
from tests.test_robotiq_communication import clocked
from tests.test_xarm_controller import config, sdk
from utils.handover_metadata import METADATA_COLUMNS, HandoverMetadataRecorder


class FixedBoxPlaceTests(unittest.TestCase):
    def args(self):
        with patch('sys.argv', ['handover.py', '--enable-follow']):
            return handover.apply_config_defaults(handover.parse_args(), config())

    def shared(self, args):
        shared = handover.FollowSharedState(args)
        shared.fixed_orientation_base = (0, 0, 0)
        shared.initial_pose_base = (.4, 0, .3, 0, 0, 0)
        return shared

    def test_teach_xyz_selects_exact_tcp_and_requires_no_geometry_samples(self):
        args = self.args()
        self.assertEqual(args.place_object_xyz_mm, (600., -300., 300.))
        self.assertFalse(args.apply_place_grasp_offset_xy)
        shared = self.shared(args)
        shared.grasp_offset_xyz_mm = np.array([50., 60., 70.])
        shared.update_place_z_samples(grasp_point_base=(.6, 0, .7))
        self.assertEqual(len(shared.recent_grasp_z_mm_buffer), 0)
        result = shared.finalize_place_z_from_recent_samples()
        self.assertTrue(result['valid'])
        self.assertEqual(result['reason'], 'fixed_xyz')
        self.assertEqual(shared.frozen_place_z_mm, 300.)
        target, _ = handover.compute_place_target(shared, apply_grasp_offset_xy=True)
        np.testing.assert_array_equal(target, [600., -300., 300.])

    def test_invalid_teach_xyz_is_rejected_instead_of_clamped_or_falling_back(self):
        teach = yaml.safe_load(Path('configs/teach.yaml').read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'teach.yaml'
            settings = config()
            settings['robot']['teach_file'] = str(path)
            for value in ([600., -300.], [600., -300., float('nan')],
                          [600., -300., 801.], [100., -300., 300.], None):
                with self.subTest(value=value):
                    teach['place_object_xyz_mm'] = value
                    path.write_text(yaml.safe_dump(teach))
                    with self.assertRaises(ValueError):
                        load_teach(settings, 'configs/handover.yaml')

    def test_legacy_xy_teach_remains_supported(self):
        teach = yaml.safe_load(Path('configs/teach.yaml').read_text())
        teach.pop('place_object_xyz_mm')
        teach['place_object_xy_mm'] = [600., -300.]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'teach.yaml'
            path.write_text(yaml.safe_dump(teach))
            settings = config()
            settings['robot']['teach_file'] = str(path)
            with patch('sys.argv', ['handover.py', '--enable-follow']):
                args = handover.apply_config_defaults(handover.parse_args(), settings)
        self.assertIsNone(args.place_object_xyz_mm)

    def test_box_release_arrives_then_descends_opens_and_backs_off_then_home(self):
        args = self.args()
        shared = self.shared(args)
        arm = sdk()
        arm.robotiq_status.update(gPO=3, gOBJ=3)
        controller = XArmController(config(), sdk=arm)
        controller.connect()
        self.addCleanup(controller.close)
        def move(*pose, **kwargs):
            arm.get_position.return_value = (0, list(pose))
            return 0
        arm.set_position.side_effect = move
        tactile = SimpleNamespace(enabled=True)
        metadata = Mock()
        with clocked(), redirect_stdout(StringIO()), \
                patch.object(handover, 'capture_tactile_release_reference', side_effect=AssertionError('unexpected tactile release')), \
                patch.object(handover, 'execute_tactile_release_descent', side_effect=AssertionError('unexpected descent')), \
                patch.object(handover, 'compute_pre_release_descend_target_mm', side_effect=AssertionError('unexpected Z calculation')), \
                patch.object(handover, 'move_robot_to_home_pose', return_value=True) as home:
            self.assertTrue(handover.execute_return_and_place(controller, shared, args, tactile_manager=tactile, metadata_recorder=metadata))
        positions = [call.args[:3] for call in arm.set_position.call_args_list]
        self.assertEqual(positions, [(600., -300., 300.), (600., -300., 200.), (580., -300., 200.)])
        arm.robotiq_set_position.assert_called_once()
        self.assertEqual(arm.robotiq_set_position.call_args.args, (0,))
        calls = [call[0] for call in arm.mock_calls]
        move_indices = [i for i, name in enumerate(calls) if name == 'set_position']
        self.assertLess(move_indices[1], calls.index('robotiq_set_position'))
        self.assertGreater(move_indices[2], calls.index('robotiq_set_position'))
        home.assert_called_once()
        metadata.note_release_tcp_target.assert_called_once_with((600., -300., 200.))
        metadata.note_delivery_bottom_location.assert_not_called()
        self.assertEqual(shared.get_snapshot()['task_state'], 'DONE')

    def test_no_open_if_fixed_target_not_reached(self):
        args = self.args()
        with patch.object(handover, 'move_robot_and_wait', return_value=False), \
                patch.object(handover, 'execute_gripper_open') as opening, \
                patch.object(handover, 'move_robot_to_home_pose') as home:
            self.assertFalse(handover.execute_return_and_place(Mock(), self.shared(args), args))
        opening.assert_not_called()
        home.assert_not_called()

    def test_no_backoff_or_home_if_open_failed(self):
        args = self.args()
        with patch.object(handover, 'move_robot_and_wait', return_value=True) as move, \
                patch.object(handover, 'execute_gripper_open', return_value=False), \
                patch.object(handover, 'move_robot_to_home_pose') as home:
            self.assertFalse(handover.execute_return_and_place(Mock(), self.shared(args), args))
        self.assertEqual(move.call_count, 2)
        home.assert_not_called()

    def test_failed_descent_cannot_open_or_return_home(self):
        args = self.args()
        with patch.object(handover, 'move_robot_and_wait', side_effect=[True, False]) as move, \
                patch.object(handover, 'execute_gripper_open') as opening, \
                patch.object(handover, 'move_robot_to_home_pose') as home:
            self.assertFalse(handover.execute_return_and_place(Mock(), self.shared(args), args))
        self.assertEqual([call.kwargs['source_mode'] for call in move.call_args_list],
                         ['fixed_box_place', 'fixed_box_descend'])
        opening.assert_not_called()
        home.assert_not_called()

    def test_zero_descent_keeps_single_arrival_before_open(self):
        args = self.args()
        args.place_descend_mm = 0.
        with patch.object(handover, 'move_robot_and_wait', return_value=True) as move, \
                patch.object(handover, 'execute_gripper_open', return_value=False):
            self.assertFalse(handover.execute_return_and_place(Mock(), self.shared(args), args))
        move.assert_called_once()
        self.assertEqual(move.call_args.args[1], (.6, -.3, .3))

    def test_invalid_or_out_of_workspace_descent_rejected_at_load(self):
        teach = yaml.safe_load(Path('configs/teach.yaml').read_text())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'teach.yaml'
            settings = config()
            settings['robot']['teach_file'] = str(path)
            for descend in (-1., float('nan'), 221.):
                with self.subTest(descend=descend):
                    teach['place_descend_mm'] = descend
                    path.write_text(yaml.safe_dump(teach))
                    with self.assertRaises(ValueError):
                        load_teach(settings, 'configs/handover.yaml')

    def test_csv_upgrade_preserves_existing_bottom_and_records_tcp_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metadata.csv'
            old_columns = [c for c in METADATA_COLUMNS if not c.startswith('release_tcp_target_')]
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=old_columns)
                writer.writeheader()
                writer.writerow({'delivery_location_est_z_mm': '80'})
            recorder = HandoverMetadataRecorder(path)
            recorder.mark_task_ready(self.shared(self.args()))
            recorder.note_release_tcp_target((600., -300., 300.))
            recorder.record_completion()
            with path.open(newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['delivery_location_est_z_mm'], '80')
            self.assertEqual(rows[0]['release_tcp_target_z_mm'], '')
            self.assertEqual(rows[1]['delivery_location_est_z_mm'], '')
            self.assertEqual(float(rows[1]['release_tcp_target_z_mm']), 300.)
            self.assertTrue(all(None not in row for row in rows))
