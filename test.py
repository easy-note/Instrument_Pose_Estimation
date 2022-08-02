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

import warmup_scheduler
import warnings

import os
import numpy as np
import random 
from tqdm import tqdm 



class TestInstruemntPoseEstimation():
    def __init__(self, configs):
        self.configs = configs

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

        # set optimiation / scheduler
        self.record_set()
        self.loss_function()

        # instrument parsing
        self.post_processing = Post_Processing(configs['dataset']['num_parts'],  configs['dataset']['num_connections'])

        
        
    def record_set(self):
        # record training process
        self.savePath, self.date_method, self.save_model_path  = save_path(self.configs)

        self.test_logger = Logger(os.path.join(self.savePath, self.date_method, 'val.log'),
                                        ['loss', 'metric'])

        

    def loss_function(self):

        self.losses = get_losses(self.configs['loss'])
        self.activation = get_activation(self.configs['loss'])
        self.metric = instrument_pose_metric(self.configs)


    def testing(self, epoch):
        self.model.eval()

        self.val_losses = AverageMeter()
        # LeftClasperPoint, RightClasperPoint, HeadPoint, ShaftPoint, EndPoint
        self.leftclasper, self.rightclasper, self.head, self.shaft, self.end = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
        # self.val_scores = AverageMeter()

        N_count = 0          
        

        for batch_idx, (images, labels) in enumerate(self.train_loader):

            images = images.cuda()
            labels = [i.cuda() for i in labels]
            N_count+= images.size(0)
    

            self.optimizer.zero_grad()

            outputs  = self.model(images)
            loss = 0
            for i in range(len(outputs)):
                if self.activation is not None:
                    outputs[i] = self.activation(outputs[i])
                loss += self.losses[i](outputs[i], labels[i]) # detection loss + regression loss

                
            self.val_losses.update(loss.item(), images.size()[0])

            parsing = self.post_processing.run(outputs[-1].detach().cpu().numpy())
            target_parsing = self.post_processing.run(labels[-1].detach().cpu().numpy())
      
            step_score = self.metric.forward(parsing, target_parsing)["F1"]
        
            leftclasper, rightclasper, head, shaft, end = step_score
            
            self.leftclasper.update(leftclasper, 1)    
            self.rightclasper.update(rightclasper, 1)    
            self.head.update(head, 1)    
            self.shaft.update(shaft, 1)    
            self.end.update(end, 1)    
            # self.val_scores.update(step_score, 1)        
        
            loss.backward()
            self.optimizer.step()


        print('Val Epoch: {} \tLoss: {:.6f}, F1 left: {:.2f}%\t rigth: {:.2f}%\t head: {:.2f}%\t shaft: {:.2f}%\t end: {:.2f}%'.format(
                epoch, self.val_losses.avg, self.leftclasper.avg, self.rightclasper.avg, self.head.avg, self.shaft.avg, self.end.avg))#self.val_scores.avg))
   
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

if __name__ == '__main__':
    args = parser.parse_args()

    cfg = Config.fromfile(args.configs)

    # cfg.merge_from_dict(args.cfg_options)
    
    IPE = TestInstruemntPoseEstimation(cfg)
    IPE.run()