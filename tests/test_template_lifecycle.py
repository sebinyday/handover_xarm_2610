from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import tempfile
import unittest
import numpy as np
from perception.sam3d_live_runtime import Sam3DLiveRuntime
from perception.sam3d_runtime import validate_template_points


class TemplateTests(unittest.TestCase):
    def make_runtime(self):
        pipeline={'object_worker_cam0':Mock(),'object_worker_cam1':Mock()}
        return Sam3DLiveRuntime(pipeline,'configs/handover.yaml')
    def test_readiness_requires_consecutive_silhouette_fits(self):
        runtime=self.make_runtime();self.addCleanup(runtime.close)
        runtime.regeneration_state='fitting'
        good=SimpleNamespace(valid=True,initialized=True,silhouette_valid_camera_count=1)
        bad=SimpleNamespace(valid=False,initialized=True,silhouette_valid_camera_count=1)
        for _ in range(runtime.required-1):self.assertFalse(runtime.after_shape_fit(good))
        runtime.after_shape_fit(bad)
        self.assertFalse(runtime.ready)
        for _ in range(runtime.required):runtime.after_shape_fit(good)
        self.assertTrue(runtime.ready)
        runtime.begin_regeneration(5)
        self.assertFalse(runtime.ready)
    def test_obsolete_generation_result_cannot_install(self):
        runtime=self.make_runtime();self.addCleanup(runtime.close)
        future=Future();future.set_result(Path('/old/template.npy'))
        runtime.future=future;runtime.future_generation=runtime.generation
        runtime.invalidate()
        with patch('perception.sam3d_live_runtime.ShapeFittingTracker') as cls:
            self.assertIsNone(runtime.poll_regeneration());cls.assert_not_called()
        self.assertFalse(runtime.ready)
    def test_generation_failure_keeps_gate_closed(self):
        runtime=self.make_runtime();self.addCleanup(runtime.close)
        future=Future();future.set_exception(RuntimeError('model failed'))
        runtime.future=future;runtime.future_generation=runtime.generation
        self.assertEqual(runtime.poll_regeneration(),'failed')
        self.assertTrue(runtime.debug().gate_blocked)
    def test_template_validation_rejects_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'template.npy';np.save(path,np.full((100,3),np.nan))
            with self.assertRaises(ValueError):validate_template_points(path,min_template_points=100)
    def test_capture_uses_same_hub_frame_and_waits_for_stability(self):
        runtime=self.make_runtime();self.addCleanup(runtime.close)
        runtime.config['runtime']['stable_capture']['stable_frames']=2
        image=np.zeros((32,32,3),np.uint8);mask=np.ones((32,32),bool)
        observation=SimpleNamespace(valid=True,frame_id=1,mask=mask,object_id=1,timestamp=1)
        runtime.pipeline['object_worker_cam0'].observation=observation
        snapshot=SimpleNamespace(pair_index=1,cam0=SimpleNamespace(color_image=image))
        with tempfile.TemporaryDirectory() as directory:
            runtime.config['runtime']['sam3d_server']['artifact_root']=directory
            runtime.capture_regeneration_frame(snapshot)
            self.assertIsNone(runtime.future)
            with patch.object(runtime,'_generate',return_value=Path(directory)/'template.npy'):
                observation.frame_id=2;snapshot.pair_index=2
                runtime.capture_regeneration_frame(snapshot)
                self.assertIsNotNone(runtime.future)
                self.assertEqual(len(list(Path(directory).glob('*/image.png'))),1)

    def test_prompt_correction_restarts_fit_with_existing_template(self):
        runtime=self.make_runtime();self.addCleanup(runtime.close)
        workers=[runtime.pipeline[f'object_worker_cam{i}'] for i in range(2)]
        for worker in workers:worker.tracker.initialized=True
        runtime.pipeline['transform_chain']=Mock()
        tracker=Mock();runtime.pipeline['shape_fitting_tracker']=tracker
        runtime.pipeline['runtime_template_path']='existing.npy'
        runtime.task_id=1;runtime.ready=True;runtime.regeneration_state='tracking';runtime.selection_requested=True
        snapshot=SimpleNamespace(cam0=Mock(),cam1=Mock(),pair_index=2)
        with patch('perception.sam3d_live_runtime.select_points') as select:
            self.assertTrue(runtime.before_frame(snapshot,task_id=1))
            self.assertEqual(select.call_count,2)
        self.assertEqual(runtime.regeneration_state,'fitting')
        tracker.reset.assert_called_once()

    def test_one_failed_camera_does_not_block_healthy_camera(self):
        runtime=self.make_runtime();self.addCleanup(runtime.close)
        runtime.pipeline['object_worker_cam0'].tracker.initialized=False
        runtime.pipeline['object_worker_cam1'].tracker.initialized=True
        runtime.pipeline['transform_chain']=Mock()
        runtime.task_id=1;runtime.ready=True;runtime.regeneration_state='tracking'
        snapshot=SimpleNamespace(cam0=Mock(),cam1=Mock(),pair_index=2)
        with patch('perception.sam3d_live_runtime.select_points') as select:
            self.assertFalse(runtime.before_frame(snapshot,task_id=1))
            select.assert_not_called()

    def test_task_change_invalidates_completed_future_before_poll(self):
        runtime=self.make_runtime();self.addCleanup(runtime.close)
        runtime.task_id=1
        future=Future();future.set_result(Path('/old/template.npy'))
        runtime.future=future;runtime.future_generation=runtime.generation
        runtime.on_task_change(2)
        with patch('perception.sam3d_live_runtime.ShapeFittingTracker') as tracker:
            self.assertIsNone(runtime.poll_regeneration())
            tracker.assert_not_called()
