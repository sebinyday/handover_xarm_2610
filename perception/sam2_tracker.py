"""Live single-object tracking using SAM2VideoPredictor's temporal memory.

Adapter targets the per-object inference-state API in the official SAM 2.1
predictor. No video files, image-only tracking, or unbounded frame lists.
"""
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import cv2
import numpy as np


def _sam2_progress(*args, **kwargs):
    """Hide the per-frame video progress bar without redirecting stderr."""
    from tqdm import tqdm
    if kwargs.get('desc') == 'propagate in video':
        kwargs['disable'] = True
    return tqdm(*args, **kwargs)


@dataclass
class MaskObservation:
    mask: np.ndarray
    camera_id: int
    frame_id: int
    timestamp: float
    valid: bool
    object_id: int = 1
    infer_ms: float = 0.0
    reason: str = ''


class SAM2Tracker:
    def __init__(self, camera_id, config, predictor=None):
        self.camera_id = int(camera_id)
        self.config = config
        self.predictor = predictor
        self.min_pixels = int(config.get('min_mask_pixels',300))
        self.history = int(config.get('max_history_frames',64))
        self.reset()

    def reset(self):
        self.state = None
        self.index = -1
        self.last_error = None
        self.last_result = None
        self.last_frame_id = -1

    @property
    def initialized(self):
        return self.state is not None

    def _load(self):
        if self.predictor is None:
            repo = Path(self.config['repo_path']).expanduser().resolve()
            if not repo.is_dir(): raise FileNotFoundError(f'SAM2 checkout missing: {repo}')
            sys.path.insert(0,str(repo))
            from sam2.build_sam import build_sam2_video_predictor
            from sam2 import sam2_video_predictor
            sam2_video_predictor.tqdm = _sam2_progress
            checkpoint = Path(self.config['checkpoint']).expanduser().resolve()
            if not checkpoint.is_file(): raise FileNotFoundError(f'SAM2 checkpoint missing: {checkpoint}')
            self.predictor = build_sam2_video_predictor(self.config['model_config'],str(checkpoint),device=self.config.get('device','cuda'))
        for name in ('_get_image_feature','add_new_points_or_box','propagate_in_video'):
            if not hasattr(self.predictor,name): raise RuntimeError(f'Unsupported SAM2 predictor: missing {name}')
        p = self.predictor
        required = max(int(getattr(p,'max_obj_ptrs_in_encoder',16)), int(getattr(p,'num_maskmem',7))*int(getattr(p,'memory_temporal_stride_for_eval',1))) + 2
        if self.history < required: raise ValueError(f'max_history_frames must be >= {required}')

    def _context(self):
        import torch
        device = str(self.predictor.device)
        return torch.autocast('cuda',dtype=torch.bfloat16) if device.startswith('cuda') and torch.cuda.is_bf16_supported() else nullcontext()

    def _image_tensor(self, frame):
        import torch
        rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image = cv2.resize(rgb,(self.predictor.image_size,self.predictor.image_size))
        tensor = torch.from_numpy(image).to(device=self.predictor.device,dtype=torch.float32).permute(2,0,1)/255.0
        mean = tensor.new_tensor([0.485,0.456,0.406])[:,None,None]
        std = tensor.new_tensor([0.229,0.224,0.225])[:,None,None]
        return (tensor-mean)/std

    def initialize(self, frame, points, labels, *, frame_id=0, timestamp=None):
        import torch
        self._load()
        points = np.asarray(points,dtype=np.float32).reshape(-1,2)
        labels = np.asarray(labels,dtype=np.int32).reshape(-1)
        h,w = frame.shape[:2]
        if len(points)!=len(labels) or not np.any(labels==1) or not np.isin(labels,[0,1]).all() or not np.isfinite(points).all() or np.any(points<0) or np.any(points[:,0]>=w) or np.any(points[:,1]>=h):
            raise ValueError('Prompt requires in-frame points and at least one positive label')
        self.reset()
        self.index = 0
        device = self.predictor.device
        state = dict(images={0:self._image_tensor(frame)},num_frames=1,offload_video_to_cpu=False,offload_state_to_cpu=False,video_height=h,video_width=w,device=device,storage_device=device,point_inputs_per_obj={},mask_inputs_per_obj={},cached_features={},constants={},obj_id_to_idx=OrderedDict(),obj_idx_to_id=OrderedDict(),obj_ids=[],output_dict_per_obj={},temp_output_dict_per_obj={},frames_tracked_per_obj={})
        self.state = state
        started = time.perf_counter()
        try:
            with torch.inference_mode(), self._context():
                self.predictor._get_image_feature(state,frame_idx=0,batch_size=1)
                _,_,masks = self.predictor.add_new_points_or_box(state,frame_idx=0,obj_id=1,points=points,labels=labels)
                # Consolidate click outputs and encode the conditioning memory.
                self.predictor.propagate_in_video_preflight(state)
            return self._result(masks,frame_id,timestamp,started)
        except BaseException:
            self.reset()
            raise

    def track(self, frame, *, frame_id, timestamp=None):
        import torch
        if self.state is None:
            return self._invalid(frame,frame_id,timestamp,'point prompt required')
        if frame_id <= self.last_frame_id:
            raise ValueError('Tracking frames must have strictly increasing frame_id')
        if frame.shape[:2] != (self.state['video_height'],self.state['video_width']):
            self.reset()
            return self._invalid(frame,frame_id,timestamp,'stream size changed; prompt again')
        started = time.perf_counter()
        self.index += 1
        state = self.state
        state['images'][self.index] = self._image_tensor(frame)
        state['num_frames'] = self.index+1
        try:
            with torch.inference_mode(), self._context():
                outputs = self.predictor.propagate_in_video(state,start_frame_idx=self.index,max_frame_num_to_track=0)
                _,_,masks = next(outputs)
                outputs.close()
            result = self._result(masks,frame_id,timestamp,started)
            self._prune()
            return result
        except Exception as exc:
            self.last_error = str(exc)
            self.reset()
            return self._invalid(frame,frame_id,timestamp,str(exc))

    def _result(self, masks, frame_id, timestamp, started):
        mask = (masks[0,0]>0).detach().cpu().numpy().astype(bool)
        valid = int(mask.sum()) >= self.min_pixels
        # Per-object presence logit rejects invisible objects, independently of mask size.
        output = self.state['output_dict_per_obj'].get(0,{})
        current = output.get('non_cond_frame_outputs',{}).get(self.index) or output.get('cond_frame_outputs',{}).get(self.index,{})
        score = current.get('object_score_logits')
        if score is not None: valid = valid and float(score.detach().cpu().reshape(-1)[0]) > 0
        result = MaskObservation(mask,self.camera_id,int(frame_id),time.time() if timestamp is None else float(timestamp),bool(valid),infer_ms=(time.perf_counter()-started)*1000,reason='' if valid else 'object absent or mask too small')
        self.last_result = result
        self.last_frame_id = int(frame_id)
        return result

    def _invalid(self,frame,frame_id,timestamp,reason):
        self.last_error = reason
        result = MaskObservation(np.zeros(frame.shape[:2],bool),self.camera_id,frame_id,time.time() if timestamp is None else timestamp,False,reason=reason)
        self.last_result = result
        return result

    def _prune(self):
        state = self.state
        cutoff = self.index-self.history+1
        # Only the latest RGB tensor is needed: older memories already contain features.
        state['images'] = {self.index:state['images'][self.index]}
        state['cached_features'] = {k:v for k,v in state['cached_features'].items() if k==self.index}
        for obj in state['output_dict_per_obj'].values():
            for key in list(obj['non_cond_frame_outputs']):
                if key < cutoff: del obj['non_cond_frame_outputs'][key]
        for group in ('frames_tracked_per_obj','point_inputs_per_obj','mask_inputs_per_obj'):
            for obj in state[group].values():
                for key in list(obj):
                    if key < cutoff and key != 0: del obj[key]
