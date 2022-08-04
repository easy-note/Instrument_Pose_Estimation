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



class InstruemntPoseEstimation():
    def __init__(self, configs):
        configs['results'] = '/raid/results/optimal_surgery/detection_only'

        self.configs = configs

        # Detect devices
        use_cuda = torch.cuda.is_available()                   # check if GPU exists
        self.device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

        # defualt setting
        self.epochs = configs['optimization']['epochs']
            
        self.train_loader, self.valid_loader, _ = get_dataloaders(configs)
        self.batch_size = configs['dataset']['batch_size']

        # model load
        self.model = model_dict[configs['model']['method']](configs=configs)
   
        # set optimiation / scheduler
        self.optimizer, self.scheduler = get_optimizer(configs, self.model)
        self.record_set()
        self.loss_function()

        # instrument parsing
        self.post_processing = Post_Processing(configs['dataset']['num_parts'],  configs['dataset']['num_connections'])

        
        
    def record_set(self):
        # record training process
        self.savePath, self.date_method, self.save_model_path  = save_path(self.configs)
        self.train_logger = Logger(os.path.join(self.savePath, self.date_method, 'train.log'),
                                        ['epoch', 'loss', 'lr'])
        self.val_logger = Logger(os.path.join(self.savePath, self.date_method, 'val.log'),
                                        ['epoch', 'loss', 'metric', 'best_metric', 'lr'])

        self.writer = SummaryWriter(os.path.join(self.savePath, self.date_method,'logfile'))
        


    def loss_function(self):

        self.losses = get_losses(self.configs['loss']) # self.losses :  [BCELoss(), MSELoss()]
        self.activation = get_activation(self.configs['loss'])
        self.metric = instrument_pose_metric(self.configs)


    def train(self, epoch):
        self.model.train()
    
        self.train_losses = AverageMeter()

        N_count = 0              

        for batch_idx, (images, seg_labels, regress_labels) in enumerate(self.train_loader):
            images = images.cuda()
            seg_labels = [i.cuda() for i in seg_labels]
            regress_labels = [i.cuda() for i in regress_labels]
            N_count+= images.size(0)
    
            self.optimizer.zero_grad()

            seg_outputs, regress_outputs = self.model(images) # [data1, data2, data3, ..., data batch]
            # print('detect_outputs : ', seg_outputs[0].shape) # torch.Size([32, 9, 320, 256])
            # print('regress_outputs : ', regress_outputs[0].shape) # torch.Size([32, 9, 320, 256])

            loss = 0
            for i in range(len(regress_outputs)): # len(regression_outputs) = 1
                if self.activation is not None:
                    seg_outputs[i] = self.activation(seg_outputs[i])

                # print('seg_labels[i] : ', seg_labels[i].shape)
                # print('regress_labels[i] : ', regress_labels[i].shape)

                loss += self.losses[0](seg_outputs[i], seg_labels[i]) # detection loss + regression loss
                loss += self.losses[1](regress_outputs[i], regress_labels[i]) # detection loss + regression loss
                # loss += self.losses[i](regress_outputs[i], regress_labels[i]) # detection loss + regression loss

                # print(loss)

                # exit(0)


            self.train_losses.update(loss.item(), images.size()[0])

        
            loss.backward()
            self.optimizer.step()

            if (batch_idx) % 10 == 0:      
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, N_count, len(self.train_loader.dataset), 100. * (batch_idx + 1) / len(self.train_loader), self.train_losses.avg))
    
    def validation(self, epoch):
        self.model.eval()

        self.val_losses = AverageMeter()
        # LeftClasperPoint, RightClasperPoint, HeadPoint, ShaftPoint, EndPoint
        self.leftclasper, self.rightclasper, self.head, self.shaft, self.end = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
        # self.val_scores = AverageMeter()

        N_count = 0          
        

        for batch_idx, (images, seg_labels, regress_labels) in enumerate(self.valid_loader): # self.train_loader -> self.valid_loader

            images = images.cuda()
            seg_labels = [i.cuda() for i in seg_labels]
            regress_labels = [i.cuda() for i in regress_labels]
            N_count+= images.size(0)
    
            self.optimizer.zero_grad()

            seg_outputs, regress_outputs  = self.model(images)
            loss = 0
            for i in range(len(regress_outputs)):
                if self.activation is not None:    
                    seg_outputs[i] = self.activation(seg_outputs[i])
                
                heatmap = regress_outputs[i].clone()

                loss += self.losses[0](seg_outputs[i], seg_labels[i]) # detection loss + regression loss
                loss += self.losses[1](regress_outputs[i], regress_labels[i]) # detection loss + regression loss

            self.val_losses.update(loss.item(), images.size()[0])

            parsing = self.post_processing.run(heatmap.detach().cpu().numpy())
            # target_parsing = self.post_processing.run(labels[-1].detach().cpu().numpy())
            target_parsing = self.post_processing.run(regress_outputs[-1].detach().cpu().numpy())
      
            step_score = self.metric.forward(parsing, target_parsing)["F1"]
            # print(step_score)
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
   
    def fit(self):
  

        self.model.to(self.device)
        # start training
        best_acc = 0
        for epoch in range(self.epochs):
            # train, test model
            self.train(epoch)
            train_losses = self.train_losses
            self.validation(epoch)
            test_losses, test_scores = self.val_losses, (self.leftclasper.avg, self.rightclasper.avg, self.head.avg, self.shaft.avg, self.end.avg)
            # if self.scheduler is not None:
            #     self.scheduler.step()
            
            # plot average of each epoch loss value
            self.train_logger.log({
                            'epoch': epoch,
                            'loss': train_losses.avg,
                            'lr': self.optimizer.param_groups[0]['lr']
                        })
            if best_acc < np.nanmean(test_scores):
                best_acc = np.nanmean(test_scores)#test_scores.avg
                torch.save({'state_dict': self.model.state_dict()}, os.path.join(self.save_model_path, 'student_best.pth'))
            self.val_logger.log({
                            'epoch': epoch,
                            'loss': test_losses.avg,
                            'metric': np.nanmean(test_scores), #test_scores.avg,
                            'best_metric' : best_acc,
                            'lr': self.optimizer.param_groups[0]['lr']
                        })
            self.writer.add_scalar('Loss/train', train_losses.avg, epoch)
            self.writer.add_scalar('Loss/test', test_losses.avg, epoch)
            
         
            self.writer.add_scalar('scores/test',np.nanmean(test_scores),  epoch) #test_scores.avg,
            torch.save({'state_dict': self.model.state_dict()}, os.path.join(self.save_model_path, 'student_lastest.pth'))  # save spatial_encoder
       

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
    
    IPE = InstruemntPoseEstimation(cfg)
    IPE.fit()