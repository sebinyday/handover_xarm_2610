import unittest
import numpy as np
import torch
from perception.sam2_tracker import SAM2Tracker


class Predictor:
    device=torch.device('cpu')
    image_size=32
    num_maskmem=3
    max_obj_ptrs_in_encoder=3
    def _get_image_feature(self,state,frame_idx,batch_size):
        state['cached_features'][frame_idx]=state['images'][frame_idx]
    def add_new_points_or_box(self,state,frame_idx,obj_id,points,labels):
        state['obj_ids']=[1]
        state['output_dict_per_obj'][0]={'cond_frame_outputs':{0:{'object_score_logits':torch.ones(1)}},'non_cond_frame_outputs':{}}
        state['frames_tracked_per_obj'][0]={}
        return 0,[1],torch.ones((1,1,state['video_height'],state['video_width']))
    def propagate_in_video_preflight(self,state):pass
    def propagate_in_video(self,state,start_frame_idx,max_frame_num_to_track):
        # A previous temporal output must survive between track calls.
        previous=state['output_dict_per_obj'][0]['non_cond_frame_outputs']
        if start_frame_idx>1:assert start_frame_idx-1 in previous
        previous[start_frame_idx]={'object_score_logits':torch.ones(1)}
        state['frames_tracked_per_obj'][0][start_frame_idx]={'reverse':False}
        self._get_image_feature(state,start_frame_idx,1)
        yield start_frame_idx,[1],torch.ones((1,1,state['video_height'],state['video_width']))


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker=SAM2Tracker(1,{'min_mask_pixels':1,'max_history_frames':8},predictor=Predictor())
        self.frame=np.zeros((24,40,3),np.uint8)
    def test_temporal_memory_bounded_for_long_stream(self):
        first=self.tracker.initialize(self.frame,[[10,10]],[1],frame_id=5,timestamp=7)
        self.assertEqual(first.mask.shape,(24,40));self.assertEqual(first.timestamp,7)
        for i in range(6,1006):self.tracker.track(self.frame,frame_id=i,timestamp=i)
        self.assertEqual(len(self.tracker.state['images']),1)
        self.assertLessEqual(len(self.tracker.state['cached_features']),1)
        self.assertLessEqual(len(self.tracker.state['output_dict_per_obj'][0]['non_cond_frame_outputs']),8)
        self.assertEqual(len(self.tracker.state['output_dict_per_obj'][0]['cond_frame_outputs']),1)
        self.assertLessEqual(len(self.tracker.state['frames_tracked_per_obj'][0]),8)
    def test_reset_requires_new_points(self):
        self.tracker.initialize(self.frame,[[10,10]],[1])
        self.tracker.reset()
        self.assertFalse(self.tracker.track(self.frame,frame_id=2).valid)
    def test_repeated_frame_rejected(self):
        self.tracker.initialize(self.frame,[[10,10]],[1],frame_id=8)
        with self.assertRaises(ValueError):self.tracker.track(self.frame,frame_id=8)
    def test_resolution_change_invalidates_state(self):
        self.tracker.initialize(self.frame,[[10,10]],[1])
        result=self.tracker.track(np.zeros((10,10,3),np.uint8),frame_id=1)
        self.assertFalse(result.valid);self.assertFalse(self.tracker.initialized)
    def test_prompts_require_positive_in_frame_point(self):
        for points,labels in [([[10,10]],[0]),([[90,10]],[1]),([[float('nan'),1]],[1])]:
            with self.assertRaises(ValueError):self.tracker.initialize(self.frame,points,labels)
