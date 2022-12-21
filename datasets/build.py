from torch.utils import data
from nlcityflowdataset import NLCityFlowDataset
from transforms import build_transforms


def build_dataset(cfg, transforms, is_training):
    dataset = NLCityFlowDataset(json_path = cfg['json_path'],
                                frame_path = cfg['frame_path'],
                                motion_path = cfg['motion_path'],
                                transforms = transforms,
                                is_training = is_training)
    
    return dataset


def make_data_loader(cfg, is_training):
    dataset = build_dataset(cfg, build_transforms(cfg, is_training), is_training)
    data_loader = data.DataLoader(dataset,
                                  batch_size = cfg['batch_size'],
                                  shuffle = is_training,
                                  num_workers = cfg['num_workers'],
                                  pin_memory = True)
    
    return data_loader