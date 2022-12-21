import torch
import torchvision.transforms as transforms


def build_transforms(cfg, is_training):
    
    if is_training:
        transforms = transforms.Compose([
            transforms.RandomContrast(0.5, 1.5),
            transforms.Resize(cfg['resize']),
            transforms.ToTensor(),
        ])
    else:
        transforms = transforms.Compose([
            transforms.Resize(cfg['resize']),
            transforms.ToTensor(),
        ])
        
    return transforms