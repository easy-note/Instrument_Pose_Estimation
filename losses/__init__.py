

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

    activation_list = []
    for method in configs['activation']:
        if method == 'sigmoid':
            activation_list.append(nn.Sigmoid())
        elif configs['activation'] == 'softmax':    
            activation_list.append(nn.Softmax())
        else:
            activation_list.append(None)


    return activation_list