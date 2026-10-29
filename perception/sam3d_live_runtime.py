"""SAM2 point selection, asynchronous template generation and fit readiness gate."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import json
import time
import uuid
import cv2
import numpy as np
import yaml
from perception.sam3d_runtime import generate_runtime_template,validate_template_points
from perception.shape_fitting_tracker_v2 import ShapeFittingTracker


def select_points(worker,bundle,frame_id):
    """Freeze RGB and matched depth while editing the prompt. Enter confirms."""
    name=f'cam{worker.camera_id}: select same object (Enter=accept, C=clear, Esc=quit)'
    image=bundle.color_image.copy()
    points=[];labels=[];preview=None;dirty=False
    def click(event,x,y,flags,param):
        nonlocal dirty
        if event in (cv2.EVENT_LBUTTONDOWN,cv2.EVENT_RBUTTONDOWN):
            points.append([x,y]);labels.append(1 if event==cv2.EVENT_LBUTTONDOWN else 0);dirty=True
    cv2.namedWindow(name)
    cv2.setMouseCallback(name,click)
    try:
        while True:
            if dirty and 1 in labels:
                preview=worker.tracker.initialize(image,points,labels,frame_id=frame_id,timestamp=bundle.timestamp_ms/1000)
                dirty=False
            display=image.copy()
            if preview is not None:
                active=preview.mask
                display[active]=(display[active]*0.55+np.array([0,220,0])*0.45).astype(np.uint8)
            for point,label in zip(points,labels):cv2.circle(display,tuple(point),5,(0,255,0) if label else (0,0,255),-1)
            cv2.imshow(name,display)
            key=cv2.waitKey(20)&0xff
            if key in (27,ord('q')):raise KeyboardInterrupt('Point selection cancelled')
            if cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE)<1: raise KeyboardInterrupt('Point selection closed')
            if key in (ord('c'),ord('C')):
                points.clear();labels.clear();preview=None;worker.tracker.reset();dirty=False
            if key in (10,13) and preview is not None and preview.valid:
                worker.prepared=preview
                return
    finally:
        cv2.destroyWindow(name)


class Sam3DLiveRuntime:
    def __init__(self,pipeline,config_path,**kwargs):
        self.pipeline=pipeline
        self.config_path=str(Path(config_path).resolve())
        self.config=yaml.safe_load(Path(config_path).read_text())
        self.task_id=None
        self.generation=0
        self.future=None
        self.future_generation=None
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='sam3d-template')
        self.regeneration_state='capturing'
        self.last_error=None
        self.ready_count=0
        self.required=max(1,int(self.config['runtime'].get('ready_valid_frames',4)))
        self.ready=False
        self.stable=[]
        self.last_pair=None
        self._force_gate=False
        self._closed=False
        self._last_seen=time.monotonic()
        self.selection_requested=False
        self.capture_allowed=True
        self._refit_fresh_pair_pending=False

    def set_capture_allowed(self,allowed):
        self.capture_allowed=bool(allowed)
        if not self.capture_allowed:
            self.last_pair=None
            self.stable.clear()

    def needs_fresh_pair(self):
        if not self.capture_allowed:
            return False
        if self.selection_requested:
            return True
        if self._refit_fresh_pair_pending:
            return True
        missing=sum(not self.pipeline[f'object_worker_cam{i}'].tracker.initialized for i in range(2))
        if missing==1 and self.regeneration_state=='refitting':
            return False
        return missing==2 or (missing==1 and not self.ready)

    def start(self):
        from tools.sam3d_ipc import build_model_spec, wait_until_ready
        server = self.config['runtime']['sam3d_server']
        repo = Path(server['repo_path']).expanduser().resolve()
        specification = build_model_spec(self.config_path, repo)
        if server.get('execution_mode', 'server') == 'server':
            wait_until_ready(server['server_socket'], expected_signature=specification['signature'], ready_timeout_s=server.get('server_ready_timeout_s',120))
        elif not Path(server['python_interpreter']).is_file():
            raise FileNotFoundError('SAM3D interpreter is missing')
        # Fail before any robot connection if the SAM2 model cannot be loaded.
        model=None
        for i in range(2):
            tracker=self.pipeline[f'object_worker_cam{i}'].tracker
            tracker.predictor=model
            tracker._load()
            model=tracker.predictor
        # Shared weights, separate states; inference is serialized by the main loop.

    @property
    def silhouette_scale_frozen(self):return self.ready or self.regeneration_state=='refitting'

    @property
    def can_refit_existing_template(self):
        return bool(self.pipeline.get('runtime_template_path') and hasattr(self.pipeline['shape_fitting_tracker'],'restart_for_refit'))

    def begin_refit(self,task_id):
        """Bind the existing template to a new handover without SAM3D generation."""
        if not self.can_refit_existing_template:
            return False
        self.generation+=1
        if self.future is not None:self.future.cancel()
        self.task_id=task_id
        tracker=self.pipeline['shape_fitting_tracker']
        if self.ready or self.regeneration_state=='refitting':
            calibrated=tracker.restart_for_refit()
        else:
            # A generated shape is not yet a verified metric template. H must
            # not bypass the original scale/silhouette readiness requirements.
            tracker.reset()
            calibrated=False
        self.ready=False;self.ready_count=0;self._force_gate=True
        self.regeneration_state='refitting' if calibrated else 'fitting'
        self.stable.clear();self.last_pair=None;self.last_error=None
        self._last_seen=time.monotonic()
        self.selection_requested=False
        self._refit_fresh_pair_pending=True
        shared=self.pipeline.get('shared_state')
        if shared is not None:shared.clear_target(reset_prediction=True,reset_arm=True)
        for i in range(2):
            worker=self.pipeline[f'object_worker_cam{i}']
            worker.prepared=None;worker.observation=None;worker.last_debug=None
        return True

    def on_task_change(self,task_id,*,preserve_template=False):
        if self.task_id is not None and self.task_id != task_id:
            if preserve_template and self.can_refit_existing_template:
                # RobotWorker changes the shared epoch before it finishes
                # publishing HOME readiness. Do not discard the reusable model
                # in that window; targets stay gated until begin_refit + ICP.
                self.generation+=1
                if self.future is not None:self.future.cancel()
                self._force_gate=True
                self.last_pair=None;self.stable.clear()
            else:
                self.invalidate()
        self.task_id=task_id

    def before_frame(self,snapshot,*,task_id,now_ros_s=None):
        self.on_task_change(task_id)
        if not self.capture_allowed:
            return True
        self._refit_fresh_pair_pending=False
        self.last_pair=snapshot
        workers=[self.pipeline[f'object_worker_cam{i}'] for i in range(2)]
        for i,worker in enumerate(workers):
            bundle=snapshot.cam0 if i==0 else snapshot.cam1
            bundle.intrinsics=self.pipeline['transform_chain'].validate_frame(i,bundle)
        missing=[i for i,worker in enumerate(workers) if not worker.tracker.initialized]
        if len(missing)==1 and (self.ready or self.regeneration_state=='refitting'):
            missing=[]  # A healthy second camera can continue; P explicitly corrects prompts.
        if self.selection_requested:
            missing=[0,1]
            self.selection_requested=False
        if missing:
            shared=self.pipeline.get('shared_state')
            if shared is not None:shared.clear_target(reset_prediction=True,reset_arm=True)
            if self.pipeline.get('runtime_template_path'):
                if self.regeneration_state=='refitting':
                    self.pipeline['shape_fitting_tracker'].restart_for_refit()
                else:
                    self.pipeline['shape_fitting_tracker'].reset()
                    self.regeneration_state='fitting'
            elif self.regeneration_state=='generating':
                self.invalidate()
            self.ready=False;self.ready_count=0
            for i in missing:select_points(workers[i],snapshot.cam0 if i==0 else snapshot.cam1,snapshot.pair_index)
        return bool(self._force_gate or not self.ready)

    def after_object_perception(self,cam0,cam1):
        if cam0.valid or cam1.valid:self._last_seen=time.monotonic()
        else:
            limit=float(self.config['grasp']['hand_relative_fallback'].get('max_dropout_sec',3))+0.75
            if time.monotonic()-self._last_seen>limit:
                self.selection_requested=True
                return True
        return bool(self._force_gate or not self.ready)

    def after_shape_fit(self,state):
        if self.regeneration_state not in {'fitting','refitting'}:return False
        if self.regeneration_state=='refitting':
            tracker=self.pipeline['shape_fitting_tracker']
            debug=getattr(tracker,'last_debug',None)
            fitness=getattr(debug,'icp_fitness',None)
            good=bool(getattr(state,'valid',False) and getattr(state,'initialized',False)
                      and getattr(state,'reason',None)=='ok'
                      and getattr(debug,'tracking_mode',None)=='translation_icp'
                      and fitness is not None and np.isfinite(fitness)
                      and fitness>=tracker.icp_min_fitness)
        else:
            good=bool(getattr(state,'valid',False) and getattr(state,'initialized',False) and getattr(state,'silhouette_valid_camera_count',0)>0)
        self.ready_count=self.ready_count+1 if good else 0
        if self.ready_count>=self.required:
            self.ready=True;self._force_gate=False;self.regeneration_state='tracking'
            return True
        return False

    def capture_regeneration_frame(self,snapshot):
        if not self.capture_allowed or self.regeneration_state!='capturing' or self.future is not None:return
        worker=self.pipeline['object_worker_cam0'];obs=worker.observation
        cfg=self.config['runtime']['stable_capture']
        if obs is None or not obs.valid or obs.frame_id!=snapshot.pair_index or obs.mask.sum()<int(cfg.get('min_mask_pixels',300)):
            self.stable.clear();return
        mask=obs.mask
        if self.stable:
            previous=self.stable[-1][1]
            iou=np.logical_and(mask,previous).sum()/max(np.logical_or(mask,previous).sum(),1)
            if iou<float(cfg.get('min_temporal_iou',0.8)):self.stable.clear()
        self.stable.append((snapshot.cam0.color_image.copy(),mask.copy()))
        count=int(cfg.get('stable_frames',3))
        self.stable=self.stable[-count:]
        if len(self.stable)<count:return
        image,mask=self.stable[-1]
        cfg_server=self.config['runtime']['sam3d_server']
        root=Path(cfg_server['artifact_root']).expanduser().resolve()
        directory=root/f'task-{self.generation}-{uuid.uuid4().hex}'
        directory.mkdir(parents=True)
        if not cv2.imwrite(str(directory/'image.png'),image) or not cv2.imwrite(str(directory/'mask.png'),mask.astype(np.uint8)*255):raise OSError('Unable to save SAM3D capture')
        (directory/'metadata.yaml').write_text(yaml.safe_dump({'backend':'sam2','object_id':obs.object_id,'camera_id':0,'frame_id':obs.frame_id,'timestamp':obs.timestamp,'generation':self.generation,'artifacts':{'image':str(directory/'image.png'),'mask':str(directory/'mask.png')}}))
        self.future_generation=self.generation
        self.future=self.executor.submit(self._generate,directory)
        self.regeneration_state='generating'

    def _generate(self,directory):
        cfg=self.config['runtime']['sam3d_server']
        path=generate_runtime_template(config_path=self.config_path,artifact_dir=directory,repo_root=Path(__file__).resolve().parents[1],execution_mode=cfg.get('execution_mode','server'),sam3d_python=cfg['python_interpreter'],sam3d_repo=cfg['repo_path'],server_socket=cfg['server_socket'],server_ready_timeout_s=cfg.get('server_ready_timeout_s',120),request_timeout_s=cfg.get('request_timeout_s',120))
        validate_template_points(path, min_template_points=int(self.config['perception']['shape_fitting']['sam3d'].get('min_template_points',100)))
        return path

    def poll_regeneration(self):
        if self.future is None or not self.future.done():return None
        future=self.future;token=self.future_generation
        self.future=None
        if self._closed or token!=self.generation:
            # Consume failures from obsolete work without installing it.
            try:future.result()
            except Exception:pass
            return None
        try:
            path=future.result()
            cfg=self.config['perception']['shape_fitting']['sam3d']
            override={'label':self.config['perception']['object']['label'],'template_id':cfg.get('template_id','sam3d_runtime'),'asset_path':str(path),'unit_scale_m':1.0,'scale_mode':cfg.get('scale_mode','axis_xyz'),'z_rotation':cfg.get('z_rotation',{}),'axis_rotation_candidates':cfg.get('axis_rotation_candidates',{})}
            tracker=ShapeFittingTracker(self.config,config_path=self.config_path,runtime_template_override=override)
            self.pipeline['shape_fitting_tracker']=tracker
            self.pipeline['runtime_template_path']=str(path)
            self.regeneration_state='fitting';self.ready=False;self.ready_count=0;self.last_error=None
            return 'installed'
        except Exception as exc:
            self.last_error=str(exc);self.ready=False;self._force_gate=True;self.regeneration_state='failed'
            return 'failed'

    def invalidate(self):
        self.generation+=1
        if self.future is not None:self.future.cancel()
        self.ready=False;self.ready_count=0;self.stable.clear();self._force_gate=True
        from pipeline import PendingTemplateTracker
        self.pipeline['shape_fitting_tracker']=PendingTemplateTracker()
        self.pipeline['runtime_template_path']=None
        self.regeneration_state='capturing'
        self._refit_fresh_pair_pending=False
        self._last_seen=time.monotonic()

    def begin_regeneration(self,task_id):
        self.task_id=task_id
        self.invalidate()
        for i in range(2):self.pipeline[f'object_worker_cam{i}'].reset()
        return True

    def debug(self):
        return SimpleNamespace(active=self.ready and self.capture_allowed,healthy=self.last_error is None,gate_blocked=not self.capture_allowed or not self.ready or self._force_gate,initialization_phase=self.regeneration_state if self.capture_allowed else 'waiting_for_robot_home',scaling_valid_frames=self.ready_count,scaling_required_frames=self.required,silhouette_scale_frozen=self.ready,last_error=self.last_error)

    def close(self):
        self._closed=True
        self.invalidate()
        self.executor.shutdown(wait=True,cancel_futures=True)
        for i in range(2):self.pipeline[f'object_worker_cam{i}'].tracker.reset()


def enforce_sam3d_target_gate(shared_state,blocked):
    if blocked:shared_state.clear_target(reset_prediction=True,reset_arm=True)
    return bool(blocked)
