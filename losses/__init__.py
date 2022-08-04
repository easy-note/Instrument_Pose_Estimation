

import torch.nn as nn

class Losses():
    def __init__(self, configs):
        self.method = configs['method']
        self.weight = configs['weight'] 
        self.reduction = configs['reduction']
        self.label_smoothing = configs['label_smoothing']

    def set_loss(self, idx):
        if self.method[idx] == 'crossentropy':
            loss = nn.CrossEntropyLoss(weight=self.weight[idx], label_smoothing=self.label_smoothing[idx], reduction=self.reduction)
        elif self.method[idx] == 'bce':
            loss = nn.BCELoss(weight=self.weight[idx], reduction=self.reduction) #, size_average=None, reduce=None, reduction='mean')
        elif self.method[idx] == 'mse':
            loss = nn.MSELoss()

        return loss
        
    def select_loss(self):
        return [self.set_loss(idx) for idx in range(len(self.method))]



def get_losses(configs):
    loss = Losses(configs)
    
    return loss.select_loss()


def get_activation(configs):
    if configs['activation'] == 'sigmoid':
        activation = nn.Sigmoid()
    elif configs['activation'] == 'softmax':    
        activation = nn.Softmax()
    else:
        activation = None


    return activation