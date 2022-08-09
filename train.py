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
        configs['results'] = '/raid/users/cv_ljh_0/instrument_pose/models'

        self.configs = configs

        # print(self.configs['configs']['nms']['windows'])
        # exit(0)

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

        import cv2           

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            
            '''
            # image & GT 비교
            print(type(images))
            print(images.shape)
            print(type(labels))
            print(labels[0].shape)

            img = images[0,:,:,:].detach().cpu().numpy()
            la = labels[0][0,:,:,:].detach().cpu().numpy()

            img = np.transpose(img, (2,1,0))
            la = np.transpose(la, (2,1,0))

            cv2.imwrite("test-1.png", img[:,:,0]+la[:,:,0]*255)
            cv2.imwrite("test-2.png", img[:,:,0]+la[:,:,1]*255)
            cv2.imwrite("test-3.png", img[:,:,0]+la[:,:,2]*255)
            cv2.imwrite("test-4.png", img[:,:,0]+la[:,:,3]*255)
            cv2.imwrite("test-5.png", img[:,:,0]+la[:,:,4]*255)
            cv2.imwrite("test-6.png", img[:,:,0]+la[:,:,5]*255)
            cv2.imwrite("test-7.png", img[:,:,0]+la[:,:,6]*255)
            cv2.imwrite("test-8.png", img[:,:,0]+la[:,:,7]*255)
            cv2.imwrite("test-9.png", img[:,:,0]+la[:,:,8]*255)
            
            exit(0)
            '''

            images = images.cuda()
            labels = [i.cuda() for i in labels]
            N_count += images.size(0)

            self.optimizer.zero_grad()

            outputs = self.model(images) # [[torch.Size([32, 9, 320, 256])], [torch.Size([32, 9, 320, 256])]]
            
            loss = 0
            for i in range(len(outputs)): # len(outputs) = 2
                if self.activation[i] is not None:
                    outputs[i] = self.activation[i](outputs[i])

                loss += self.losses[i](outputs[i], labels[i]) # detection loss + regression loss

                # print(loss)

            self.train_losses.update(loss.item(), images.size()[0])
        
            loss.backward()
            self.optimizer.step()

            if (batch_idx) % 10 == 0:      
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, N_count, len(self.train_loader.dataset), 100. * (batch_idx + 1) / len(self.train_loader), \
                    self.train_losses.avg))
    
    
    def validation(self, epoch):
        self.model.eval()

        self.val_losses = AverageMeter()
        # LeftClasperPoint, RightClasperPoint, HeadPoint, ShaftPoint, EndPoint
        self.leftclasper, self.rightclasper, self.head, self.shaft, self.end = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
        # self.val_scores = AverageMeter()

        N_count = 0          
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.valid_loader): # self.train_loader -> self.valid_loader

                images = images.cuda()
                labels = [i.cuda() for i in labels]
                N_count+= images.size(0)
        
                # self.optimizer.zero_grad()
                
                outputs  = self.model(images)
                loss = 0
                for i in range(len(outputs)): # len(outputs) = 2
                    if self.activation[i] is not None:
                        heatmap = outputs[-1].clone() # regression
                        outputs[i] = self.activation[i](outputs[i])
                    
                    loss += self.losses[i](outputs[i], labels[i]) # detection loss + regression loss
                
                self.val_losses.update(loss.item(), images.size()[0])   
                        
                # parsing = self.post_processing.run(heatmap.detach().cpu().numpy()) # prediction
                parsing = self.post_processing.run(heatmap.detach().cpu().numpy(), self.configs['configs']['nms']['window'])
                target_parsing = self.post_processing.run(labels[-1].detach().cpu().numpy(), 200) # regression gt : 정답이 json 형태 -> list 형태로 변환해야 함. 

                step_score = self.metric.forward(parsing, target_parsing)["F1"]
                # print(step_score)
                leftclasper, rightclasper, head, shaft, end = step_score
                
                self.leftclasper.update(leftclasper, 1)    
                self.rightclasper.update(rightclasper, 1)    
                self.head.update(head, 1)    
                self.shaft.update(shaft, 1)    
                self.end.update(end, 1)    
                
                # self.val_scores.update(step_score, 1)        
                # loss.requires_grad_(True)
                # loss.backward()
                # self.optimizer.step()


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
                torch.save({'state_dict': self.model.state_dict()}, os.path.join(self.save_model_path, 'model_best.pth'))
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
            print(self.save_model_path)
            torch.save({'state_dict': self.model.state_dict()}, os.path.join(self.save_model_path, 'model_lastest.pth'))  # save spatial_encoder
       

import argparse
from mmcv import Config

parser = argparse.ArgumentParser()
parser.add_argument(
    'configs',
    default='./dataset/', type=str, help='Root of directory path of data'
    )

if __name__ == '__main__':
    import os
    import cv2

    args = parser.parse_args()

    cfg = Config.fromfile(args.configs)

    # cfg.merge_from_dict(args.cfg_options)
    
    IPE = InstruemntPoseEstimation(cfg)
    IPE.fit()