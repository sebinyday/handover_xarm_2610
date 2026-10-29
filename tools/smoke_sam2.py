#!/usr/bin/env python3
"""Check real point initialization and temporal tracking on synthetic frames."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from perception.sam2_tracker import SAM2Tracker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/handover.yaml')
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--frames',type=int,default=3)
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())['perception']['object']['sam2'].copy()
    if args.cpu:
        import torch
        torch.set_num_threads(2)
        config['device']='cpu'
    config['min_mask_pixels']=1
    first=SAM2Tracker(0,config)
    image=np.zeros((128,128,3),np.uint8);image[30:100,40:90]=[210,90,20]
    initial=first.initialize(image,[[60,60]],[1],frame_id=0,timestamp=0)
    assert initial.valid and initial.mask.shape==image.shape[:2]
    second=SAM2Tracker(1,config,predictor=first.predictor)
    second.initialize(image,[[60,60]],[1],frame_id=0,timestamp=0)
    for i in range(1,args.frames):
        for tracker in (first,second):
            result=tracker.track(image,frame_id=i,timestamp=i)
            assert result.valid and result.frame_id==i and result.mask.shape==image.shape[:2]
            assert len(tracker.state['images'])==1
    print(f'SAM2 smoke passed: {args.frames} frames per camera, shared weights, independent temporal states')


if __name__=='__main__':main()
