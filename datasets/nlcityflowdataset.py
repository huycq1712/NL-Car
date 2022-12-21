import os
import json
import random
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import re


class NLCityFlowDataset(Dataset):
    
    def __init__(self, json_path, frame_path, motion_path, transforms, is_training) -> None:
        super().__init__()
        self.json_path = json_path
        self.frame_path = frame_path
        self.motion_path = motion_path
        self.transforms = transforms
        self.is_training = is_training
        
        with open(json_path, 'r') as f:
            self.tracks = json.load(f)
            
        self.list_of_tracks = list(self.tracks.values())
        self.list_of_uuids = list(self.tracks.keys())
        
        self.all_indexs = list(range(len(self.list_of_uuids)))
        #self.flip_tags = [False] * len(self.list_of_uuids)
        
        
    def __len__(self):
        return len(self.list_of_uuids)
    
    
    def __getitem__(self, index):
        index = self.all_indexs[index]
        track = self.list_of_tracks[index]
        #flag = self.flip_tags[index]
        
        nl_idx = int(random.uniform(0, len(track['nl'])))
        nl_view_idx = int(random.uniform(0, len(track['nl_other'])))
        frame_idx = int(random.uniform(0, len(track['frames'])))
        
        nl = track['nl'][nl_idx]
        if len(track['nl_other']) == 0:
            nl_view = track['nl_other'][-1]
        else:
            nl_view = track['nl_other'][nl_view_idx]
            
        car_noun = 'this is a' + extract_np(nl)
        
        frame = Image.open(os.path.join(self.frame_path, track['frames'][frame_idx]))
        motion = Image.open(os.path.join(self.motion_path, track['frames'][frame_idx]))
        bbox = track['bbox'][frame_idx]
        bbox = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        crop = frame.crop(bbox)
        
        if self.transformes:
            crop = self.transforms(crop)
        
        return {
            'index': index,
            'crop_image': crop,
            'global_image': frame,
            'motion_image': motion,
            'nl': nl,
            'nl_view': nl_view,
            'car_noun': car_noun,
        }
        
        
def extract_np(nl):
    return nl
            
        
        
        
    