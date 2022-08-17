# Task Transfer Learning
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from models import model_dict

from tools.metric import save_path, Logger, AverageMeter
from tools.post_processing import Post_Processing
from tools.visualization import Visualization

import warmup_scheduler
import warnings

import os
import numpy as np
import random 
from tqdm import tqdm 
from PIL import Image

import cv2
import albumentations as A
import albumentations.pytorch as AP

import matplotlib.pyplot as plt
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
class InferenceInstruemntPoseEstimation():
    def __init__(self, configs, args):
        self.cfg = configs
        
        self.instrument_name = ['LeftClasper', 'RightClasper', 'Head', 'Shaft', 'End', 'Left_Head', 'Right_Head', 'Head_Shaft', 'Shaft_End']
        # Detect devices
        use_cuda = torch.cuda.is_available()                   # check if GPU exists
        self.device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
        # Dataload
        self.datalist = configs['dataset']['images']
        if self.datalist is None:
            self.datalist = [args.img_path]
        
        
        # model load
        self.model = model_dict[configs['model']['method']](configs=configs)
        checkpoint = torch.load(configs['model']['checkpoint'])
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)


        # instrument parsing
        self.post_processing = Post_Processing(configs['dataset']['num_parts'],  configs['dataset']['num_connections'])
        self.visualization = Visualization(dest_path = configs['dest_path'])

        self.root = configs['dataset']['root']
       
        self.mean = self.cfg['dataset']['normalization']['mean']
        self.std = self.cfg['dataset']['normalization']['std']
        self.unnorm = UnNormalize(self.mean, self.std)

        self.transforms = A.Compose(self.get_transforms(configs))
    def get_transforms(self, configs):
        width, height = configs['dataset']['img_size']
        trans = [A.Resize(width=width, height=height )]


        trans.append(A.Normalize(mean=self.mean,
                                     std=self.std))
        trans.append(AP.transforms.ToTensorV2())
        return trans
    def NormalizeData(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    def heatmap(self, image, pred, filename):

        save = os.path.join(self.cfg['dest_path'], 'heatmap')
        os.makedirs(save, exist_ok=True)

        detection_list = pred[0].squeeze().detach().cpu().numpy()
        regression_list = pred[1].squeeze().detach().cpu().numpy()
        for i in range(9):
            detection = detection_list[i]
            regression = regression_list[i]

            plt.subplot(1,4,1)
            plt.imshow(image)
            plt.xlabel('Image')

            plt.subplot(1,4,2)
            plt.imshow(self.NormalizeData(detection), cmap='gray', vmin=0, vmax=1)
            plt.xlabel('Detection output')
            

            plt.subplot(1,4, 3)
            plt.imshow(self.NormalizeData(regression), cmap='gray', vmin=0, vmax=1)
            plt.xlabel('Regression output')
            
            plt.subplot(1,4, 4)
            regression = cv2.GaussianBlur(regression, (100+1, 100+1), 0)
            plt.imshow(self.NormalizeData(regression), cmap='gray', vmin=0, vmax=1)
            plt.xlabel('Gaussian filter')

            plt.savefig(os.path.join(save, '{}_{}.png'.format(filename, self.instrument_name[i])))
            plt.close()
    def imageopen(self, image_path):
        return cv2.cvtColor(np.array(Image.open(image_path)), cv2.COLOR_RGB2BGR)


    def inference(self):
        self.model.eval()
        
        for filename in self.datalist:
            image = self.imageopen(os.path.join(self.root, filename))
            image = self.transforms(image=image)['image'].to(self.device).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(image)
                parsing = self.post_processing.run(outputs[-1].detach().cpu().numpy(), self.cfg['nms']['window'])
                image = cv2.cvtColor(self.unnorm(image.squeeze()).permute(1,2,0).detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
                self.visualization.show(image, parsing, filename.split('/')[-1])
                self.heatmap(image, outputs, filename.split('/')[-1])
        

import argparse
from mmcv import Config

parser = argparse.ArgumentParser()
parser.add_argument(
    'configs',
    default='./dataset/', type=str, help='Root of directory path of data'
    )
parser.add_argument(
    '--img_path',
    default='/raid/datasets/public/EndoVisPose/Training/training/image/img_000255_raw_train3.jpg', type=str, help='Root of directory path of data'
    )

if __name__ == '__main__':
    args = parser.parse_args()

    cfg = Config.fromfile(args.configs)
    IPE = InferenceInstruemntPoseEstimation(cfg.configs, args)
    IPE.inference()