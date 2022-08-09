import os
from datetime import datetime

import json
import csv
from sklearn.metrics import average_precision_score
import numpy as np 
import torch

def save_path(configs):

    date = datetime.today().strftime("%Y%m%d%H%M") 
    dataset = configs['dataset']['dataset']
    model = configs['model']['method']
    date_method = os.path.join(dataset, model, date)
    if 'results' in configs.keys():
        print( configs['results'])
    else:
        for key in configs:
            print(key, configs[key])
    result_path = configs['results']
            
    if not os.path.exists(os.path.join(result_path, date_method)):
        os.makedirs(os.path.join(result_path, date_method))
    save_model_path = os.path.join(result_path, date_method, 'models')
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    
    if not os.path.exists(os.path.join(result_path, date_method,'logfile')):
        os.makedirs(os.path.join(result_path, date_method,'logfile'))
    
    return result_path, date_method, save_model_path

class AverageMeter(object):
    """
    Computes and stores the average and
    current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SegAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()




if __name__ == '__main__':
    cifar2ann()