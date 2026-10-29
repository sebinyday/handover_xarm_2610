"""Reuse a generated object template without camera/robot/model connections."""

import tempfile
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import handover
from perception.sam3d_live_runtime import Sam3DLiveRuntime
from perception.shape_fitting_tracker_v2 import ShapeFittingTracker
from pipeline import PendingTemplateTracker
from system.shared_state import HandState, MergedObjectState, ObjectState
from tests.test_xarm_controller import config
from utils.realsense_stream import FrameBundle


class SameObjectRefitTests(unittest.TestCase):
    def runtime(self):
        fit = SimpleNamespace(
            restart_for_refit=Mock(return_value=True), reset=Mock(),
            last_debug=SimpleNamespace(icp_fitness=.9, tracking_mode='translation_icp'),
            icp_min_fitness=.02,
        )
        pipeline = {
            'runtime_template_path': 'existing-template.npy',
            'shape_fitting_tracker': fit,
            'transform_chain': SimpleNamespace(validate_frame=lambda i, frame: frame.intrinsics),
        }
        for i in range(2):
            pipeline[f'object_worker_cam{i}'] = SimpleNamespace(
                tracker=SimpleNamespace(initialized=True, reset=Mock()),
                reset=Mock(), prepared=object(), observation=object(), last_debug=object(),
            )
        runtime = Sam3DLiveRuntime(pipeline, 'configs/handover.yaml')
        runtime.task_id = 3
        runtime.ready = True
        runtime.regeneration_state = 'tracking'
        self.addCleanup(runtime.close)
        return runtime, pipeline, fit

    def snapshot(self, number=1):
        frame = FrameBundle(
            np.zeros((32, 32, 3), np.uint8), np.ones((32, 32), np.float32),
            {'fx':100., 'fy':100., 'cx':16., 'cy':16.}, time.time() * 1000, 'test',
        )
        return SimpleNamespace(cam0=frame, cam1=frame, pair_index=number, timestamp_delta_ms=0., within_sync_tolerance=True)

    def test_new_epoch_retains_template_and_sam2_memory_without_prompts(self):
        runtime, pipeline, fit = self.runtime()
        trackers = [pipeline[f'object_worker_cam{i}'].tracker for i in range(2)]
        with patch.object(runtime, '_generate') as generate, patch('perception.sam3d_live_runtime.select_points') as select:
            self.assertTrue(runtime.begin_refit(4))
            self.assertTrue(runtime.needs_fresh_pair())
            runtime.on_task_change(4)
            self.assertTrue(runtime.before_frame(self.snapshot(), task_id=4))
            runtime.capture_regeneration_frame(self.snapshot())
        self.assertIs(pipeline['shape_fitting_tracker'], fit)
        self.assertEqual(pipeline['runtime_template_path'], 'existing-template.npy')
        self.assertEqual(runtime.task_id, 4)
        self.assertEqual(runtime.regeneration_state, 'refitting')
        self.assertFalse(runtime.ready)
        self.assertTrue(runtime.silhouette_scale_frozen)
        self.assertFalse(runtime.needs_fresh_pair())
        select.assert_not_called()
        generate.assert_not_called()
        fit.reset.assert_not_called()
        for tracker in trackers:
            tracker.reset.assert_not_called()

    def test_refit_requires_consecutive_good_icp_not_old_or_held_geometry(self):
        runtime, _, fit = self.runtime()
        runtime.begin_refit(4)
        good = SimpleNamespace(valid=True, initialized=True, reason='ok', silhouette_valid_camera_count=0)
        for _ in range(runtime.required - 1):
            self.assertFalse(runtime.after_shape_fit(good))
        fit.last_debug.icp_fitness = 0.
        self.assertFalse(runtime.after_shape_fit(good))
        self.assertEqual(runtime.ready_count, 0)
        fit.last_debug.icp_fitness = .9
        fit.last_debug.tracking_mode = 'hold'
        self.assertFalse(runtime.after_shape_fit(good))
        fit.last_debug.tracking_mode = 'translation_icp'
        for _ in range(runtime.required):
            runtime.after_shape_fit(good)
        self.assertTrue(runtime.ready)
        self.assertFalse(runtime.debug().gate_blocked)

    def test_late_generation_cannot_replace_template_during_refit(self):
        runtime, pipeline, fit = self.runtime()
        future = Future()
        future.set_result(Path('/obsolete-template.npy'))
        runtime.future, runtime.future_generation = future, runtime.generation
        runtime.begin_refit(4)
        with patch('perception.sam3d_live_runtime.ShapeFittingTracker') as build:
            self.assertIsNone(runtime.poll_regeneration())
        build.assert_not_called()
        self.assertIs(pipeline['shape_fitting_tracker'], fit)

    def test_lost_sam2_views_need_only_prompts_and_keep_calibrated_template(self):
        runtime, pipeline, fit = self.runtime()
        runtime.begin_refit(4)
        for i in range(2):
            pipeline[f'object_worker_cam{i}'].tracker.initialized = False
        with patch('perception.sam3d_live_runtime.select_points') as select:
            runtime.before_frame(self.snapshot(), task_id=4)
        self.assertEqual(select.call_count, 2)
        fit.reset.assert_not_called()
        self.assertEqual(runtime.regeneration_state, 'refitting')
        self.assertEqual(pipeline['runtime_template_path'], 'existing-template.npy')

    def test_one_lost_view_can_continue_with_the_other_during_refit(self):
        runtime, pipeline, _ = self.runtime()
        runtime.begin_refit(4)
        pipeline['object_worker_cam0'].tracker.initialized = False
        with patch('perception.sam3d_live_runtime.select_points') as select:
            runtime.before_frame(self.snapshot(), task_id=4)
        select.assert_not_called()
        self.assertFalse(runtime.needs_fresh_pair())

    def test_missing_template_cannot_start_or_destroy_current_tracking(self):
        runtime, pipeline, fit = self.runtime()
        pipeline['runtime_template_path'] = None
        self.assertFalse(runtime.begin_refit(4))
        self.assertEqual(runtime.task_id, 3)
        self.assertTrue(runtime.ready)
        fit.restart_for_refit.assert_not_called()

    def test_unfinished_initial_scale_reuses_template_but_finishes_normal_fitting(self):
        runtime, pipeline, fit = self.runtime()
        fit.restart_for_refit.return_value = False
        runtime.begin_refit(4)
        self.assertEqual(runtime.regeneration_state, 'fitting')
        self.assertFalse(runtime.silhouette_scale_frozen)
        self.assertEqual(pipeline['runtime_template_path'], 'existing-template.npy')

    def test_h_cannot_skip_initial_silhouette_readiness_for_unverified_scale(self):
        runtime, pipeline, fit = self.runtime()
        runtime.ready, runtime.regeneration_state = False, 'fitting'
        runtime.begin_refit(4)
        fit.restart_for_refit.assert_not_called()
        fit.reset.assert_called_once()
        self.assertEqual(runtime.regeneration_state, 'fitting')
        self.assertFalse(runtime.silhouette_scale_frozen)
        self.assertEqual(pipeline['runtime_template_path'], 'existing-template.npy')

    def test_real_icp_reseeds_far_position_without_reinitializing_scale_or_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'template.npy'
            rng = np.random.default_rng(10)
            canonical = rng.uniform(-.5, .5, (600, 3)) * np.array([.10, .07, .05])
            np.save(path, canonical)
            settings = {'perception': {'shape_fitting': {
                'enabled':True, 'template_library':{'object':{'asset_path':str(path), 'unit_scale_m':1., 'scale_mode':'axis_xyz'}},
                'cluster':{'dbscan_eps_m':.04, 'dbscan_min_points':2},
                'scale_init':{'stable_frames':1, 'percentile_low':0., 'percentile_high':100., 'min_scale':.5, 'max_scale':2.},
                'icp':{'max_points':1000, 'max_iterations':10, 'distance_threshold_m':.15, 'crop':{'enabled':False}},
                'downsample':{'voxel_size_m':0., 'max_points':1000},
                'silhouette_constraint':{'enabled':True, 'apply_to_all':True},
            }}}
            fit = ShapeFittingTracker(settings)
            def cloud(points):
                return MergedObjectState(valid=True, label='object', merged_points_base=points.tolist())
            first = fit.process(cloud(canonical + [.45, 0, .2]), freeze_silhouette_scale=True)
            self.assertTrue(first.valid)
            scale, rotation = fit._frozen_scale_xyz.copy(), fit._frozen_rotation.copy()
            delta = np.array([.4, -.3, .25])
            moved = np.asarray(first.fitted_points_base) + delta
            self.assertTrue(fit.restart_for_refit())
            with patch.object(fit, '_initialize_template', side_effect=AssertionError('scale/rotation reinitialized')), patch('perception.shape_fitting_tracker_v2.compute_silhouette_score', side_effect=AssertionError('scale search restarted')):
                second = fit.process(cloud(moved), freeze_silhouette_scale=True)
            self.assertTrue(second.valid)
            np.testing.assert_allclose(second.centroid_base, np.asarray(first.centroid_base) + delta, atol=1e-6)
            np.testing.assert_array_equal(fit._frozen_scale_xyz, scale)
            np.testing.assert_array_equal(fit._frozen_rotation, rotation)
            self.assertFalse(fit._translation_reseed_pending)

    def exercise_h_key(self, robot_enabled, delayed_home=False):
        settings = config()
        snapshot = self.snapshot()
        hub = Mock()
        hub.read_next_pair.return_value = snapshot
        hub.read_fresh_pair.return_value = snapshot
        fit = PendingTemplateTracker()
        fit.reset = Mock()
        fit.restart_for_refit = Mock(return_value=True)
        pipeline = {
            'sensor_hub':hub, 'shape_fitting_tracker':fit, 'runtime_template_path':'cached.npy',
            'prompt_classes':['object'], 'object_backend':'sam3d', 'ablation_mode':'baseline',
            't_cam0_base':np.eye(4), 't_cam1_base':np.eye(4),
            'transform_chain':SimpleNamespace(validate_frame=lambda i, frame: frame.intrinsics, t_base_cam0=np.eye(4), t_base_cam1=np.eye(4)),
        }
        for i in range(2):
            worker = Mock()
            worker.tracker.initialized = True
            worker.process_frame.return_value = ObjectState(camera_id=i)
            worker.last_debug = None
            pipeline[f'object_worker_cam{i}'] = worker
            hand = Mock()
            hand.process_frame.return_value = HandState(camera_id=i)
            pipeline[f'hand_worker_cam{i}'] = hand
        for name, cls in [
            ('hand_selector', handover.HandSelector), ('object_merger', handover.ObjectMerger),
            ('fusion', handover.PerceptionFusion), ('grasp_planner', handover.GraspTargetPlanner),
            ('grasp_z_stabilizer', handover.GraspPointZStabilizer), ('hand_relative_fallback', handover.HandRelativeFallbackTracker),
        ]:
            pipeline[name] = cls(settings)
        runtime = Sam3DLiveRuntime(pipeline, 'configs/handover.yaml')
        runtime.task_id, runtime.ready, runtime.regeneration_state = 0, True, 'tracking'
        self.addCleanup(runtime.close)
        status = handover.RobotStatus(state=handover.ROBOT_STATE_FOLLOWING, home_ready=True)
        robot = Mock()
        robot.get_status.side_effect = lambda: status
        events = []
        def submit(request):
            if request.type == handover.ROBOT_REQ_RESET_HOME:
                pipeline['shared_state'].reset_for_restart(follow_enabled=True)
                if delayed_home:
                    status.state, status.home_ready = handover.ROBOT_STATE_RESETTING, False
                else:
                    events.append('HOME')
                    status.reset_done_epoch += 1
                    status.task_ready_epoch += 1
            return 1
        robot.submit.side_effect = submit
        def fresh(**kwargs):
            if robot_enabled:
                self.assertEqual(events, ['HOME'])
            events.append('fresh')
            return snapshot
        hub.read_fresh_pair.side_effect = fresh
        ui_ticks = 0
        def key(_):
            nonlocal ui_ticks
            ui_ticks += 1
            if ui_ticks == 1:
                return ord('h')
            if ui_ticks == 2:
                if delayed_home:
                    self.assertEqual(pipeline['runtime_template_path'], 'cached.npy')
                    self.assertIs(pipeline['shape_fitting_tracker'], fit)
                    self.assertEqual(runtime.task_id, 1)
                    fit.restart_for_refit.assert_not_called()
                    events.append('HOME')
                    status.state, status.home_ready = handover.ROBOT_STATE_FOLLOWING, True
                    status.reset_done_epoch += 1
                    status.task_ready_epoch += 1
                return -1
            return ord('q')
        with patch('sys.argv', ['handover.py'] + (['--enable-follow'] if robot_enabled else [])), patch.object(handover, 'build_dual_perception_pipeline', return_value=pipeline), patch.object(handover, 'Sam3DLiveRuntime', return_value=runtime), patch.object(runtime, 'start'), patch.object(runtime, 'close'), patch.object(runtime, '_generate') as generate, patch('perception.sam3d_live_runtime.select_points') as select, patch.object(handover, 'RobotWorker', return_value=robot), patch.object(handover, 'AnySkinTactileManager'), patch.object(handover.cv, 'imshow'), patch.object(handover.cv, 'destroyAllWindows'), patch.object(handover.cv, 'waitKey', side_effect=key):
            handover.main()
        self.assertIs(pipeline['shape_fitting_tracker'], fit)
        self.assertEqual(pipeline['runtime_template_path'], 'cached.npy')
        self.assertEqual(runtime.task_id, 1)
        self.assertEqual(pipeline['shared_state'].task_epoch, 1)
        self.assertIsNone(pipeline['shared_state'].get_snapshot()['latest_target_xyz_mm'])
        fit.restart_for_refit.assert_called_once()
        fit.reset.assert_not_called()
        select.assert_not_called()
        generate.assert_not_called()
        hub.read_fresh_pair.assert_called_once()
        for i in range(2):
            pipeline[f'object_worker_cam{i}'].tracker.reset.assert_not_called()

    def test_h_key_waits_for_home_then_reuses_template_in_robot_mode(self):
        self.exercise_h_key(robot_enabled=True)

    def test_h_key_reuses_template_in_perception_only_mode(self):
        self.exercise_h_key(robot_enabled=False)

    def test_h_keeps_template_when_epoch_changes_before_home_readiness_is_published(self):
        self.exercise_h_key(robot_enabled=True, delayed_home=True)


if __name__ == '__main__':
    unittest.main()
