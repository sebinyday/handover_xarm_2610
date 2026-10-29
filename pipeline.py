"""Build only the active SAM2 / MediaPipe / SAM3D perception path."""
from pathlib import Path
from types import SimpleNamespace
import yaml
from calibration_io import load_transform_chain
from perception.sam2_tracker import SAM2Tracker
from perception.object_worker import ObjectWorker
from perception.hand_worker import HandWorker
from perception.hand_selector import HandSelector
from perception.object_merger import ObjectMerger
from perception.fusion import PerceptionFusion
from perception.grasp_target import GraspTargetPlanner
from perception.grasp_z_stabilizer import GraspPointZStabilizer
from perception.hand_relative_fallback import HandRelativeFallbackTracker
from system.dual_sensor_hub import DualSensorHub
import numpy as np


class PendingTemplateTracker:
    last_debug=None
    def reset(self): pass
    def process(self,merged_object,**kwargs):
        return SimpleNamespace(valid=False,initialized=False,label=merged_object.label,centroid_base=None,fitted_points_base=[],reason='waiting_for_template',template_id=None,scale=None,scale_xyz=None,scale_mode='axis_xyz',template_axes_base=None,z_rotation_deg=None,roll_rotation_deg=None,pitch_rotation_deg=None,bowl_height_fraction=None)


def build_dual_perception_pipeline(args,*,runtime_template_path=None):
    config=yaml.safe_load(Path(args.config).read_text())
    transform=load_transform_chain(args.config)
    pipeline={'sensor_hub':DualSensorHub.from_config(args.config),'transform_chain':transform,'t_cam0_base':np.linalg.inv(transform.t_base_cam0),'t_cam1_base':np.linalg.inv(transform.t_base_cam1),'shape_fitting_tracker':PendingTemplateTracker(),'prompt_classes':[config['perception']['object']['label']],'object_backend':'sam3d','runtime_template_path':None,'ablation_mode':'baseline'}
    created=[]
    try:
        for i in range(2):
            tracker=SAM2Tracker(i,config['perception']['object']['sam2'])
            pipeline[f'object_worker_cam{i}']=ObjectWorker(i,tracker,transform,config)
            hand=HandWorker(i,transform,config)
            pipeline[f'hand_worker_cam{i}']=hand
            created.append(hand)
        for name,cls in [('hand_selector',HandSelector),('object_merger',ObjectMerger),('fusion',PerceptionFusion),('grasp_planner',GraspTargetPlanner),('grasp_z_stabilizer',GraspPointZStabilizer),('hand_relative_fallback',HandRelativeFallbackTracker)]:
            pipeline[name]=cls(config)
        return pipeline
    except BaseException:
        for hand in created: hand.close()
        raise
