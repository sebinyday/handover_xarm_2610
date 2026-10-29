import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import yaml
from calibration_io import CameraCalibration,TransformChain,load_transform_chain
from perception.object_worker import ObjectWorker
from perception.sam2_tracker import MaskObservation
from utils.realsense_stream import FrameBundle


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        K=np.array([[100,0,10],[0,100,10],[0,0,1]],float)
        matrix=np.eye(4);matrix[:3,3]=[1,2,3]
        self.chain=TransformChain([CameraCalibration('one',matrix,K,np.zeros(5)),CameraCalibration('two',np.eye(4),K,np.zeros(5))])
        self.frame=FrameBundle(np.zeros((20,20,3),np.uint8),np.ones((20,20),np.float32),dict(fx=100,fy=100,cx=10,cy=10),123000,'one')
    def test_direct_color_transform(self):
        result=self.chain.transform_points_camera_to_base(0,[[0,0,1]])
        np.testing.assert_allclose(result,[[1,2,4]])
    def test_serial_and_intrinsics_mismatch_rejected(self):
        self.frame.serial='wrong'
        with self.assertRaises(ValueError):self.chain.validate_frame(0,self.frame)
        self.frame.serial='one';self.frame.intrinsics['fx']=200
        with self.assertRaises(ValueError):self.chain.validate_frame(0,self.frame)
    def test_depth_shape_must_match_color(self):
        self.frame.depth_image_m=np.ones((4,4))
        with self.assertRaises(ValueError):self.chain.validate_frame(0,self.frame)
    def test_mask_and_depth_are_same_frame(self):
        obs=MaskObservation(np.ones((20,20),bool),0,5,123,True)
        tracker=SimpleNamespace(track=lambda *a,**kw:obs)
        config={'perception':{'object':{'label':'object','point_cloud':{'stride':1,'outlier_method':'none','min_points_per_camera':1}}}}
        worker=ObjectWorker(0,tracker,self.chain,config)
        state=worker.process_frame(self.frame,5)
        self.assertTrue(state.valid)
        self.assertEqual(state.timestamp,123)
        self.assertAlmostEqual(state.centroid_base[2],4)
        obs.frame_id=4
        with self.assertRaisesRegex(ValueError,'identity'):worker.process_frame(self.frame,5)
    def test_degenerate_transform_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.json'
            path.write_text(json.dumps({'units':'meters','serial':'one','T_base_color':np.zeros((4,4)).tolist(),'color_K':np.eye(3).tolist(),'color_D':[0]*5}))
            with self.assertRaises(ValueError):CameraCalibration.read(path)
