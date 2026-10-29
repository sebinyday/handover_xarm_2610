"""Lift a frame-matched SAM2 mask into the calibrated xArm base frame."""
from dataclasses import dataclass
import numpy as np
from object_pt_extraction.mask_types import MaskInstance
from object_pt_extraction.pointcloud_utils import build_point_cloud_from_instances, voxel_downsample_point_cloud, remove_statistical_outliers, remove_radius_outliers
from system.shared_state import ObjectState


@dataclass
class ObjectWorkerDebug:
    camera_id: int
    combined_mask: np.ndarray
    points_camera: np.ndarray
    points_base: np.ndarray
    colors_rgb: np.ndarray
    infer_ms: float


class ObjectWorker:
    def __init__(self,camera_id,tracker,transform_chain,config):
        self.camera_id=camera_id
        self.tracker=tracker
        self.transform_chain=transform_chain
        self.config=config
        self.label=str(config['perception']['object'].get('label','object'))
        self.last_debug=None
        self.observation=None
        self.prepared=None

    def reset(self):
        self.tracker.reset()
        self.last_debug=None
        self.observation=None
        self.prepared=None

    def process_frame(self,frame_bundle,frame_id=-1):
        intrinsics=self.transform_chain.validate_frame(self.camera_id,frame_bundle)
        if self.prepared is not None and self.prepared.frame_id == frame_id:
            observation=self.prepared
            self.prepared=None
        else:
            observation=self.tracker.track(frame_bundle.color_image,frame_id=frame_id,timestamp=frame_bundle.timestamp_ms/1000)
        self.observation=observation
        if observation.camera_id != self.camera_id or observation.frame_id != frame_id or observation.timestamp != frame_bundle.timestamp_ms/1000:
            raise ValueError('SAM2 mask/depth frame identity mismatch')
        cfg=self.config['perception']['object']['point_cloud']
        mask=observation.mask if observation.valid else np.zeros_like(observation.mask)
        instances=[MaskInstance(mask)] if observation.valid else []
        mask,points,_,_,colors=build_point_cloud_from_instances(frame_bundle.color_image,frame_bundle.depth_image_m,intrinsics,instances,stride=int(cfg.get('stride',2)),max_points=int(cfg.get('max_points',20000)),min_depth_m=float(cfg.get('min_depth_m',0.1)),max_depth_m=float(cfg.get('max_depth_m',1.5)))
        base=self.transform_chain.transform_points_camera_to_base(self.camera_id,points)
        if len(base):
            base,colors=voxel_downsample_point_cloud(base,colors,voxel_size_m=float(cfg.get('per_camera_voxel_size_m',0.005)))
            method=cfg.get('outlier_method','statistical')
            if method=='statistical': base,colors=remove_statistical_outliers(base,colors,nb_neighbors=int(cfg.get('outlier_nb_neighbors',20)),std_ratio=float(cfg.get('outlier_std_ratio',2)))
            elif method=='radius': base,colors=remove_radius_outliers(base,colors,radius_m=float(cfg.get('outlier_radius_m',0.01)),min_neighbors=int(cfg.get('outlier_min_neighbors',8)))
        valid=observation.valid and len(base)>=int(cfg.get('min_points_per_camera',300))
        self.last_debug=ObjectWorkerDebug(self.camera_id,mask,points,base,colors,observation.infer_ms)
        return ObjectState(camera_id=self.camera_id,frame_id=frame_id,object_detected=valid,label=self.label,confidence=1.0 if valid else 0.0,centroid_base=tuple(np.mean(base,axis=0)) if valid else None,point_count=len(base),points_base=base.tolist(),timestamp=observation.timestamp,valid=valid)
