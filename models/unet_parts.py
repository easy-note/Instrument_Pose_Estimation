""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F



class CB(nn.Module):
    """(convolution (non-stride) => [BN])"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()

        self.cb = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.cb(x)
class CBR(nn.Module):
    """(convolution (non-stride) => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        
        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.cbr(x)


class SBR(nn.Module):
    """(convolution (stride) => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        
        self.sbr = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.sbr(x)

class DBR(nn.Module):
    """(deconvolution) => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        
        self.dbr = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.dbr(x)

class CBS(nn.Module):
    """(convolution (non-stride) => [BN] => Softmax)"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super().__init__()
        
        self.cbs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.cbs(x)