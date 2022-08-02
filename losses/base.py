import torch.nn as nn

class Losses():
    def __init__(self, configs):
        self.method = configs['method']
        self.weight = configs['weight'] 
        self.label_smoothing = configs['label_smoothing']


      

    def set_loss(self, idx):
        if self.method[idx] == 'crossentropy':
            loss = nn.CrossEntropyLoss(weight=self.weight[idx], label_smoothing=self.label_smoothing[idx])
       
        elif self.method[idx] == 'mse':
            loss = nn.MSELoss()

        return loss
        
    def select_loss(self):
        return [self.set_loss(idx) for idx in range(len(self.method))]











