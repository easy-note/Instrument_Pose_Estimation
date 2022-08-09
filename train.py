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


from tools.metric import save_path, Logger, AverageMeter, SegAverageMeter
from tools.metric import instrument_pose_metric
from tools.metric import IntersectionAndUnion, Pixel_ACC
from tools.post_processing import Post_Processing

import warmup_scheduler
import warnings

import os
import numpy as np
import random 
from tqdm import tqdm 

import matplotlib.pyplot as plt

class InstruemntPoseEstimation():
    def __init__(self, configs):
        # configs['results'] = '/raid/results/optimal_surgery/detection_only'

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
        
        self.num_parts = configs['dataset']['num_parts']
        self.num_connections = configs['dataset']['num_connections']
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

        self.losses = get_losses(self.configs['loss'])
        self.activation = get_activation(self.configs['loss'])
        self.metric = instrument_pose_metric(self.configs)


    def train(self, epoch):
        self.model.train()
    

        self.train_losses = AverageMeter()
        pixel_acc_meter = SegAverageMeter()
        intersection_meter = SegAverageMeter()
        union_meter = SegAverageMeter()

        N_count = 0          
        cnt = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.cuda()
            labels = [i.cuda() for i in labels]
            N_count+= images.size(0)
    

            self.optimizer.zero_grad()

            outputs  = self.model(images)
            loss, pixel_acc, iou = 0, 0, 0
            for i in range(len(outputs)):
                if self.activation[i] is not None:
                    outputs[i] = self.activation[i](outputs[i])
                # outputs[i] = outputs[i].view(images.size(0), -1)
                # labels[i] = labels[i].view(images.size(0), -1)

                loss += self.losses[i](outputs[i], labels[i]) # detection loss + regression loss
                
                if i == 0:
                    scores = outputs[i]
                    targets = labels[i].clone().cpu().detach().numpy()
                    for j in range(self.num_connections + self.num_parts):
                        score = scores[:, j, ...]
                        pred = np.ones(score.size())
                        pred = pred * (score.clone().cpu().detach().numpy() > 0.5)
                        target = targets[:, j, ...]
             
                        pixel_acc, pix = Pixel_ACC(pred, target)
                        intersection, union = IntersectionAndUnion(pred, target, numClass=2)
                
                        pixel_acc_meter.update(pixel_acc, pix)
                        intersection_meter.update(intersection)
                        union_meter.update(union)

                        if j == 0 and cnt == 0:
                            cnt += 1
                            plt.imshow(pred[0], cmap='gray', vmin=0, vmax=1)
                   
                            plt.savefig('/raid/results/pose/train_img/pred_{}.png'.format(str(epoch)))
                            plt.close()
                            plt.imshow(target[0], cmap='gray', vmin=0, vmax=1)
                   
                            plt.savefig('/raid/results/pose/train_img/gt_{}.png'.format(str(epoch)))
                            plt.close()
            self.train_losses.update(loss.item(), images.size()[0])
        
            loss.backward()
            self.optimizer.step()

            if (batch_idx) % 10 == 0:      
                
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, N_count, len(self.train_loader.dataset), 100. * (batch_idx + 1) / len(self.train_loader), self.train_losses.avg), end=" ")
                iou = intersection_meter.sum / (union_meter.sum + 1e-10)
                print("Segment Eval Pixcel ACC{:.2f}%\t IoU {:.2f}%".format(pixel_acc_meter.average() * 100, iou.mean()))
    
    def validation(self, epoch):
        self.model.eval()

        self.val_losses = AverageMeter()
        # LeftClasperPoint, RightClasperPoint, HeadPoint, ShaftPoint, EndPoint
        self.leftclasper, self.rightclasper, self.head, self.shaft, self.end = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
        self.rmse = AverageMeter()
        # self.val_scores = AverageMeter()

        N_count = 0    

        pixel_acc_meter = SegAverageMeter()
        intersection_meter = SegAverageMeter()
        union_meter = SegAverageMeter()

        with torch.no_grad():
            cnt = 0
            for batch_idx, (images, labels) in enumerate(self.train_loader):

                images = images.cuda()
                labels = [i.cuda() for i in labels]
                N_count+= images.size(0)


                outputs  = self.model(images)
                loss = 0
                for i in range(len(outputs)):
                    if self.activation[i] is not None:
                        heatmap = outputs[i].clone()
                        outputs[i] = self.activation[i](outputs[i])
                    loss += self.losses[i](outputs[i], labels[i]) # detection loss + regression loss

                    if i == 0:
                        scores = outputs[i]
                        targets = labels[i].clone().cpu().detach().numpy()
                        for j in range(self.num_connections + self.num_parts):
                            score = scores[:, j, ...]
                            pred = np.ones(score.size())
                            pred = pred * (score.clone().cpu().detach().numpy() > 0.5)
                            target = targets[:, j, ...]
                
                            pixel_acc, pix = Pixel_ACC(pred, target)
                            intersection, union = IntersectionAndUnion(pred, target, numClass=2)
                    
                            pixel_acc_meter.update(pixel_acc, pix)
                            intersection_meter.update(intersection)
                            union_meter.update(union)
                            if j == 0 and cnt == 0:
                                plt.imshow(pred[0], cmap='gray', vmin=0, vmax=1)
                                plt.savefig('/raid/results/pose/test_img/{}.png'.format(str(epoch)))
                                plt.close()
                                plt.imshow(target[0], cmap='gray', vmin=0, vmax=1)
                   
                                plt.savefig('/raid/results/pose/test_img/gt_{}.png'.format(str(epoch)))
                                plt.close()
             
                    
            self.val_losses.update(loss.item(), images.size()[0])

            parsing = self.post_processing.run(heatmap.detach().cpu().numpy(), self.configs['nms']['window'])
            target_parsing = self.post_processing.run(labels[-1].detach().cpu().numpy(), 10)
            # print(parsing[0])
            # print(target_parsing[0])
            # print("====================")
            step_score = self.metric.forward(parsing, target_parsing)
            # print(step_score)
            leftclasper, rightclasper, head, shaft, end = step_score["F1"]
            rmse = np.nanmean(step_score['RMSE'])
            
            self.leftclasper.update(leftclasper, 1)    
            self.rightclasper.update(rightclasper, 1)    
            self.head.update(head, 1)    
            self.shaft.update(shaft, 1)    
            self.end.update(end, 1)  
            self.rmse.update(rmse, 1)  
            # self.val_scores.update(step_score, 1)        
        
            
            



        print('Val Epoch: {} \tLoss: {:.6f}, F1 left: {:.2f}%\t rigth: {:.2f}%\t head: {:.2f}%\t shaft: {:.2f}%\t end: {:.2f}%\t RMSE: {:.2f}%'.format(
                epoch, self.val_losses.avg, self.leftclasper.avg, self.rightclasper.avg, self.head.avg, self.shaft.avg, self.end.avg, self.rmse.avg), end=" ")#self.val_scores.avg))
        iou = intersection_meter.sum / (union_meter.sum + 1e-10)
        print("Segment Eval Pixcel ACC{:.2f}%\t IoU {:.2f}%".format(pixel_acc_meter.average() * 100, iou.mean()))
    def fit(self):
  

        self.model.to(self.device)
        # start training
        best_acc = 0
        for epoch in range(self.epochs):
            # train, test model
            self.train(epoch)
            train_losses = self.train_losses
            self.validation(epoch)
            test_losses, test_scores = self.val_losses, self.rmse.avg #(self.leftclasper.avg, self.rightclasper.avg, self.head.avg, self.shaft.avg, self.end.avg)

            if self.scheduler is not None:
                self.scheduler.step()
            
            # plot average of each epoch loss value
            self.train_logger.log({
                            'epoch': epoch,
                            'loss': train_losses.avg,
                            'lr': self.optimizer.param_groups[0]['lr']
                        })
            if best_acc < np.nanmean(test_scores):
                best_acc = np.nanmean(test_scores)#test_scores.avg
                torch.save({'state_dict': self.model.state_dict()}, os.path.join(self.save_model_path, 'best.pth'))
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
            torch.save({'state_dict': self.model.state_dict()}, os.path.join(self.save_model_path, 'lastest.pth'))  # save spatial_encoder
       

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
    
    IPE = InstruemntPoseEstimation(cfg.configs)
    IPE.fit()