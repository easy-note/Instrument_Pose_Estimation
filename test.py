# Task Transfer Learning
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from dataload import get_dataloaders
from models import model_dict
from losses import get_losses, get_activation
from scheduler import get_optimizer


# from dataload.augmentation import augmentation_dict


from tools.metric import save_path, Logger, AverageMeter
from tools.metric import instrument_pose_metric
from tools.post_processing import Post_Processing
from tools.visualization import Visualization

import warmup_scheduler
import warnings

import os
import cv2
import numpy as np
import random 
from tqdm import tqdm 
import math

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

class TestInstruemntPoseEstimation():
    def __init__(self, configs):
        self.configs = configs.configs

        # Detect devices
        use_cuda = torch.cuda.is_available()                   # check if GPU exists
        self.device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
            
        _, _, self.test_loader = get_dataloaders(configs)

        # model load
        self.model = model_dict[configs['model']['method']](configs=configs)
        checkpoint = torch.load(configs['model']['checkpoint'])
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.visualization = Visualization(dest_path = self.configs['dest_path'] + '/test_vis')
        self.visualization_gt = Visualization(dest_path = self.configs['dest_path'] + '/test_vis_gt')
        # set optimiation / scheduler
        self.record_set()
        self.loss_function()

        self.image_list = sorted(os.listdir(self.configs['dataset']['images_dir']['test']))
        # instrument parsing
        self.post_processing = Post_Processing(self.configs['dataset']['num_parts'],  configs['dataset']['num_connections'])
        
    def record_set(self):
    

        self.test_logger = Logger(os.path.join(self.configs['dest_path'], 'test_vis', 'perform.log'),
                                        ['left', 'right', 'head', 'shaft', 'end', 'rmse'])

        
        self.mean = self.configs['dataset']['normalization']['mean']
        self.std = self.configs['dataset']['normalization']['std']
        self.unnorm = UnNormalize(self.mean, self.std)

    def loss_function(self):
        self.losses = get_losses(self.configs['loss'])
        self.activation = get_activation(self.configs['loss'])
        self.metric = instrument_pose_metric(self.configs)


    
    def testing(self):
        self.model.eval()

        losses_meter = AverageMeter()
        # LeftClasperPoint, RightClasperPoint, HeadPoint, ShaftPoint, EndPoint
        precision_tools = [AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()]
        recall_tools = [AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()]
        rmse_tools = [AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()]
        totals = [AverageMeter(),AverageMeter(),AverageMeter()]
        
        iteration = 0    



        with torch.no_grad():

            for batch_idx, (images, labels) in enumerate(self.test_loader):

                images = images.cuda()
                labels = [i.cuda() for i in labels]




                outputs  = self.model(images)[-1].detach().cpu().numpy()
          
     
                parsing = self.post_processing.run(outputs, self.configs['nms']['window'])
                target_parsing = self.post_processing.run(labels[-1].detach().cpu().numpy(), 5)

                n = images.size(0)

                # images = self.unnorm(images).permute(0, 2,3,1).detach().cpu().numpy()

                for idx in range(n):
                    image = cv2.cvtColor(self.unnorm(images[idx]).permute(1, 2,0).detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
                    self.visualization.show(image, [parsing[idx]], self.image_list[iteration])

                    self.visualization_gt.show(image, [target_parsing[idx]], self.image_list[iteration])
                    iteration += 1

                step_score = self.metric.forward(parsing, target_parsing)
                # eftclasper rightclasper head shaft end
                precisions = step_score["Precision"]
                recalls = step_score["Recall"]
                rmses = step_score["RMSE"]

                totals[0].update(np.nanmean(precisions))
                totals[1].update(np.nanmean(recalls))
                totals[2].update(np.nanmean(rmses))
                for idx in range(len(precisions)):
                    precision = precisions[idx]
                    recall = recalls[idx]
                    rmse = rmses[idx]
                    if not math.isnan(precision):
                        precision_tools[idx].update(precision, 1)    
                    if not math.isnan(recall):
                        recall_tools[idx].update(recall, 1)    
                    if not math.isnan(rmse):
                        rmse_tools[idx].update(rmse, 1)    
                

        print("LeftClasper", "RightClasper", "Head", "Shaft", "End", "Total")
        for i in range(self.configs['dataset']['num_parts']):
            print("{:.2f} / {:.2f} / {:.2f}".format(precision_tools[i].avg, recall_tools[i].avg,  rmse_tools[i].avg), end=' | ')
        print("{:.2f} / {:.2f} / {:.2f}".format(totals[0].avg, totals[1].avg, totals[2].avg), )
    def run(self):
  
        best_acc = 0
        
        
        self.testing()
        # test_scores =  (self.leftclasper.avg, self.rightclasper.avg, self.head.avg, self.shaft.avg, self.end.avg)
        # self.test_logger.log({
                 
        #                     'metric': np.nanmean(test_scores), 
        #                 })
       

import argparse
from mmcv import Config

parser = argparse.ArgumentParser()
parser.add_argument(
    'configs',
    default='./dataset/', type=str, help='Root of directory path of data'
    )

if __name__ == '__main__':
    args = parser.parse_args()

    cfg = Config.fromfile(args.configs)

    # cfg.merge_from_dict(args.cfg_options)
    
    IPE = TestInstruemntPoseEstimation(cfg)
    IPE.run()