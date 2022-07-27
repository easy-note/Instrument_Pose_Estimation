# Task Transfer Learning
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


from models import model_dict
from losses import get_losses
from scheduler import get_optimizer


# from dataload.augmentation import augmentation_dict
from dataload import dataload_dict

from tools.metric import save_path, Logger, AverageMeter
from tools.metric import instrument_pose_metric
from tools.post_processing import Post_Processing

import warmup_scheduler
import warnings

import os
import numpy as np
import random 
from tqdm import tqdm 



class InstruemntPoseEstimation():
    def __init__(self, configs):
        self.configs = configs

        # Detect devices
        use_cuda = torch.cuda.is_available()                   # check if GPU exists
        self.device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

        # defualt setting
        self.epochs = configs['epochs']
        
        # data load
        self.train_loader, self.valid_loader = dataload_dict(configs)
        self.batch_size = configs['batch_size']

        # model load
        self.model = model_dict(configs['model'])
      
        # set optimiation / scheduler
        self.optimizer, self.scheduler = get_optimizer(configs, self.model)
        self.record_set()
        self.loss_function()

        # instrument parsing
        self.post_processing = Post_Processing(configs['dataset']['num_parts'],  configs['dataset']['num_connections'])

        
        
    def record_set(self):
        opt = self.opt
        # record training process
        self.savePath, self.date_method, self.save_model_path  = save_path(configs)
        self.train_logger = Logger(os.path.join(self.savePath, self.date_method, 'train.log'),
                                        ['epoch', 'loss', 'lr'])
        self.val_logger = Logger(os.path.join(self.savePath, self.date_method, 'val.log'),
                                        ['epoch', 'loss', 'acc', 'best_acc', 'lr'])

        self.writer = SummaryWriter(os.path.join(self.savePath, self.date_method,'logfile'))
        


    def loss_function(self):

        self.losses = loss_dict(self.configs['loss'])
        self.metric = minstrument_pose_metric(self.configs)


    def train(self, epoch):
        self.model.train()
    

        self.train_losses = AverageMeter()

        N_count = 0          
        

        for batch_idx, (images, labels, index) in enumerate(self.train_loader):
            images = images.cuda()
            labels = [i.cuda() for i in labels]
            N_count+= images.size(0)
    

            self.optimizer.zero_grad()

            outputs  = self.model(images)
            loss = 0
            for i in range(len(outputs)):
                loss += self.losses[i](outputs[i], labels[i]) # detection loss + regression loss

                
            self.train_losses.update(loss.item(), images.size()[0])

        
            loss.backward()
            self.optimizer.step()

            if (batch_idx) % 10 == 0:      
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, N_count, len(self.train_loader.dataset), 100. * (batch_idx + 1) / len(self.train_loader), self.train_losses.avg))
    
    def validation(self, epoch):
        self.model.eval()

        self.val_losses = AverageMeter()
        self.val_scores = AverageMeter()

        N_count = 0          
        

        for batch_idx, (images, labels, index) in enumerate(self.train_loader):
            images = images.cuda()
            labels = [i.cuda() for i in labels]
            N_count+= images.size(0)
    

            self.optimizer.zero_grad()

            outputs  = self.model(images)
            loss = 0
            for i in range(len(outputs)):
                loss += self.losses[i](outputs[i], labels[i]) # detection loss + regression loss

                
            self.val_losses.update(loss.item(), images.size()[0])

            parsing = self.post_processing.run(outputs[-1])
            step_score = self.metric(parsing, labels[-1].data)["RMSE"]
            self.val_scores.update(step_score,images.size()[0])        
        
            loss.backward()
            self.optimizer.step()

            if (batch_idx) % 10 == 0:      
                print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                    epoch, N_count, len(self.val_loader.dataset), 100. * (batch_idx + 1) / len(self.val_loader), self.val_losses.avg, self.val_scores.avg))
   
        return self.val_losses, self.val_scores
    def fit(self):
        
        self.model.to(self.device)
        # start training
        best_acc = 0
        for epoch in range(self.epochs):
            # train, test model
            self.train(epoch)
            train_losses = self.train_losses

            test_losses, test_scores = validation(epoch)
            self.scheduler.step()
            
            # plot average of each epoch loss value
            self.train_logger.log({
                            'epoch': epoch,
                            'loss': train_losses.avg,
                            'lr': self.optimizer.param_groups[0]['lr']
                        })
            if best_acc < test_scores.avg:
                best_acc = test_scores.avg
                torch.save({'state_dict': self.model.state_dict()}, os.path.join(self.save_model_path, 'student_best.pth'))
            self.val_logger.log({
                            'epoch': epoch,
                            'loss': test_losses.avg,
                            'metric': test_scores.avg,
                            'best_metric' : best_acc,
                            'lr': self.optimizer.param_groups[0]['lr']
                        })
            self.writer.add_scalar('Loss/train', train_losses.avg, epoch)
            self.writer.add_scalar('Loss/test', test_losses.avg, epoch)
            
            self.writer.add_scalar('scores/train', train_scores.avg, epoch)
            self.writer.add_scalar('scores/test', test_scores.avg, epoch)
            torch.save({'state_dict': self.model.state_dict()}, os.path.join(self.save_model_path, 'student_lastest.pth'))  # save spatial_encoder
       

import argparse
import mmcv

parser = argparse.ArgumentParser()
parser.add_argument(
    'configs',
    default='./dataset/', type=str, help='Root of directory path of data'
    )
args = parser.parse_args()
if __name__ == '__main__':
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)
    
    IPE = InstruemntPoseEstimation(cfg)
    IPE.fit()