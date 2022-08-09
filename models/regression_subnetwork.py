""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch

class RegressionSubnetwork(nn.Module):
    def __init__(self, configs): # (n_channels, n_classes, bilinear=False):
        super(RegressionSubnetwork, self).__init__()

        self.n_channels = configs['model']['n_channels'] # 64
        self.bilinear = configs['model']['bilinear'] # false
        self.n_classes = configs['dataset']['num_parts'] + configs['dataset']['num_connections'] # 9


        self.cbr1 = CBR(in_channels=3+9, out_channels=self.n_channels, kernel_size=3)
        self.cbr2 = CBR(in_channels=64, out_channels=128, kernel_size=3)
        self.cbr3 = CBR(in_channels=128, out_channels=256, kernel_size=3)
        self.cbr4 = CBR(in_channels=256, out_channels=256, kernel_size=3)
        self.cbr5 = CBR(in_channels=256, out_channels=256, kernel_size=1, padding=0)

        self.cb = CBR(in_channels=256, out_channels=self.n_classes, kernel_size=1, padding=0)


    def forward(self, x):

        x1 = self.cbr1(x)
        x2 = self.cbr2(x1)
        x3 = self.cbr3(x2)
        x4 = self.cbr4(x3)
        x5 = self.cbr5(x4)

        logits = self.cb(x5) # torch.Size([10, 9, 512, 512])

        return logits



def base_models(configs, **kwargs):
    print(configs)
    model = RegressionSubnetwork(configs, **kwargs)
    return model  