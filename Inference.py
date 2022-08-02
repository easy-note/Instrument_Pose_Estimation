# Task Transfer Learning
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from models import model_dict


from tools.post_processing import Post_Processing
from tools.visualization import Visualization

import warmup_scheduler
import warnings

import os
import numpy as np
import random 
from tqdm import tqdm 


import albumentations as A
import albumentations.pytorch as AP

class InferenceInstruemntPoseEstimation():
    def __init__(self, configs, args):
        self.configs = configs

        # Detect devices
        use_cuda = torch.cuda.is_available()                   # check if GPU exists
        self.device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
        # Dataload
        self.datalist = configs['dataset']['images']
        self.transforms = self.get_transforms(configs)

        # model load
        self.model = model_dict[configs['model']['method']](configs=configs)
        checkpoint = torch.load(configs['model']['checkpoint'])
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        # set optimiation / scheduler
        self.record_set()
        self.loss_function()

        # instrument parsing
        self.post_processing = Post_Processing(configs['dataset']['num_parts'],  configs['dataset']['num_connections'])
        self.visualization = Visualization(dest_path = configs['dest_path'])
        
    
    def get_transforms(self, configs):
        width, height = configs['dataset']['img_size']
        trans = [A.Resize(width=width, height=height )]

        for method in configs['dataset']['augmentation'][mode]:
            if method == 'verticalflip':
                trans.append(A.VerticalFlip(0.5))
            elif method == 'horizonflip':
                trans.append(A.HorizontalFlip(0.5))

        
        
        trans.append(A.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]))
        trans.append(AP.transforms.ToTensorV2())
        return trans


    def inference(self, images, filenames):
        self.model.eval()
        images = self.transforms(image=images)
        outputs  = self.model(images)
            

        parsing = self.post_processing.run(outputs[-1].detach().cpu().numpy())
        self.visualization.run(parsing, filenames) # visualization instrument keypoint & save filename.png
    
   
    def run(self):
  
        best_acc = 0
        
        
        self.testing(epoch)
        test_losses, test_scores = self.val_losses, (self.leftclasper.avg, self.rightclasper.avg, self.head.avg, self.shaft.avg, self.end.avg)
        self.test_logger.log({
                            'loss': test_losses.avg,
                            'metric': np.nanmean(test_scores), 
                        })
       

import argparse
from mmcv import Config

parser = argparse.ArgumentParser()
parser.add_argument(
    'configs',
    default='./dataset/', type=str, help='Root of directory path of data'
    )
parser.add_argument(
    '--img_path',
    default='/raid/datasets/public/EndoVisPose/Training/test/image/img_0250_test5.jpg', type=str, help='Root of directory path of data'
    )
if __name__ == '__main__':
    args = parser.parse_args()

    cfg = Config.fromfile(args.configs)

    # cfg.merge_from_dict(args.cfg_options)
    
    IPE = TestInstruemntPoseEstimation(cfg)
    IPE.run()