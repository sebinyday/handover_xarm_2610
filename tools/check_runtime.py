#!/usr/bin/env python3
"""Read-only readiness report; never opens cameras, serial ports or robot sockets."""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from calibration_io import load_transform_chain
from robot.xarm_controller import load_teach
from tools.sam3d_ipc import build_model_spec


def check(config_path,require_robot=False):
    path=Path(config_path).resolve()
    config=yaml.safe_load(path.read_text())
    report={'config':str(path),'checks':{},'errors':[]}
    def attempt(name,operation):
        try:
            value=operation()
            report['checks'][name]=value if value is not None else True
        except Exception as exc:
            report['checks'][name]=False;report['errors'].append(f'{name}: {exc}')
    attempt('calibration',lambda:[c.serial for c in load_transform_chain(path).cameras])
    if require_robot:attempt('teaching',lambda:load_teach(config,path) and True)
    for package in ['numpy','scipy','cv2','mediapipe','open3d','torch','torchvision','pyrealsense2','flask','hydra','iopath','pytorch3d','kaolin','spconv','moge','utils3d','gsplat','safetensors','timm']+(['xarm','anyskin'] if require_robot else []):
        if importlib.util.find_spec(package) is None:
            report['checks'][package]=False;report['errors'].append(f'{package}: package missing')
        else:report['checks'][package]=True
    sam2=config['perception']['object']['sam2']
    for name,location in [('sam2_source',Path(sam2['repo_path'])/'sam2/sam2_video_predictor.py'),('sam2_checkpoint',Path(sam2['checkpoint']))]:
        if not location.is_file():report['checks'][name]=False;report['errors'].append(f'{name}: missing {location}')
        else:report['checks'][name]=str(location)
    server=config['runtime']['sam3d_server']
    report['checks']['sam3d_same_environment'] = Path(server['python_interpreter']).resolve() == Path(sys.executable).resolve()
    report['checks']['sam3d_source'] = (Path(server['repo_path']) / 'sam3d_objects' / '__init__.py').is_file()
    attempt('sam3d_assets',lambda:build_model_spec(path,server['repo_path'])['signature'])
    if not Path(server['python_interpreter']).is_file():report['errors'].append(f"sam3d_interpreter: missing {server['python_interpreter']}")
    try:
        import torch
        cuda=bool(torch.cuda.is_available())
        report['checks']['cuda']=cuda
        report['checks']['torch_version']=torch.__version__
        if not cuda and (sam2.get('device','cuda').startswith('cuda') or config['perception']['shape_fitting']['sam3d'].get('require_cuda',True)):
            report['errors'].append('CUDA unavailable; configured SAM2/SAM3D GPU inference cannot start')
    except Exception as exc:report['errors'].append(f'torch import: {exc}')
    report['ready']=not report['errors']
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/handover.yaml')
    parser.add_argument('--robot',action='store_true',help='Also validate teach settings and robot/tactile packages, without connecting')
    args=parser.parse_args()
    result=check(args.config,args.robot)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['ready'] else 1


if __name__=='__main__':sys.exit(main())
