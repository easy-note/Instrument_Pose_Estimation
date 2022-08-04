""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import torch

class DetectionSubnetwork(nn.Module):
    def __init__(self, configs): # n_channels, n_classes, bilinear=False):
        super(DetectionSubnetwork, self).__init__()

        self.n_channels = configs['model']['n_channels']
        self.bilinear = configs['model']['bilinear']
        self.n_classes = configs['dataset']['num_parts'] + configs['dataset']['num_connections'] 
        
        n_places = self.n_channels

        self.down1 = CBR(in_channels=3, out_channels=self.n_channels, kernel_size=3)
        
        self.down2_b1_sbr = SBR(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.down2_b2_sbr = SBR(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.down2_b1_cbr = CBR(in_channels=64, out_channels=64, kernel_size=3)
        self.down2_b2_cbr = CBR(in_channels=64, out_channels=64, kernel_size=3)
        self.down2_cbr = CBR(in_channels=128, out_channels=128, kernel_size=1, padding=0)

        self.down3_b1_sbr = SBR(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.down3_b2_sbr = SBR(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.down3_b1_cbr = CBR(in_channels=128, out_channels=128, kernel_size=3)
        self.down3_b2_cbr = CBR(in_channels=128, out_channels=128, kernel_size=3)
        self.down3_cbr = CBR(in_channels=256, out_channels=256, kernel_size=1, padding=0)

        self.down4_b1_sbr = SBR(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.down4_b2_sbr = SBR(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.down4_b1_cbr = CBR(in_channels=256, out_channels=256, kernel_size=3)
        self.down4_b2_cbr = CBR(in_channels=256, out_channels=256, kernel_size=3)
        self.down4_cbr = CBR(in_channels=512, out_channels=512, kernel_size=1, padding=0)

        self.down5_b1_sbr = SBR(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.down5_b2_sbr = SBR(in_channels=512, out_channels=512, kernel_size=2, stride=2)
        self.down5_b1_cbr = CBR(in_channels=512, out_channels=512, kernel_size=3)
        self.down5_b2_cbr = CBR(in_channels=512, out_channels=512, kernel_size=3)
        self.down5_cbr = CBR(in_channels=1024, out_channels=1024, kernel_size=1, padding=0)


        self.up1_b1_dbr = DBR(in_channels=1024, out_channels=256, kernel_size=2, stride=2)
        self.up1_b2_dbr = DBR(in_channels=1024, out_channels=256, kernel_size=2, stride=2)
        self.up1_b1_cbr = CBR(in_channels=256, out_channels=256, kernel_size=3)
        self.up1_b2_cbr = CBR(in_channels=256, out_channels=256, kernel_size=3)
        self.up1_cbr = CBR(in_channels=1024, out_channels=512, kernel_size=1, padding=0)

        self.up2_b1_dbr = DBR(in_channels=512, out_channels=128, kernel_size=2, stride=2)
        self.up2_b2_dbr = DBR(in_channels=512, out_channels=128, kernel_size=2, stride=2)
        self.up2_b1_cbr = CBR(in_channels=128, out_channels=128, kernel_size=3)
        self.up2_b2_cbr = CBR(in_channels=128, out_channels=128, kernel_size=3)
        self.up2_cbr = CBR(in_channels=512, out_channels=256, kernel_size=1, padding=0)

        self.up3_b1_dbr = DBR(in_channels=256, out_channels=64, kernel_size=2, stride=2)
        self.up3_b2_dbr = DBR(in_channels=256, out_channels=64, kernel_size=2, stride=2)
        self.up3_b1_cbr = CBR(in_channels=64, out_channels=64, kernel_size=3)
        self.up3_b2_cbr = CBR(in_channels=64, out_channels=64, kernel_size=3)
        self.up3_cbr = CBR(in_channels=128, out_channels=128, kernel_size=1, padding=0)

        self.up4_b1_dbr = DBR(in_channels=128, out_channels=32, kernel_size=2, stride=2)
        self.up4_b2_dbr = DBR(in_channels=128, out_channels=32, kernel_size=2, stride=2)
        self.up4_b1_cbr = CBR(in_channels=32, out_channels=32, kernel_size=3)
        self.up4_b2_cbr = CBR(in_channels=32, out_channels=32, kernel_size=3)
        self.up4_cbr = CBR(in_channels=64, out_channels=64, kernel_size=1, padding=0)

        self.cbs = CBS(in_channels=64, out_channels=self.n_classes, kernel_size=1, padding=0)


    def forward(self, x):
        ## down sampling
        x1 = self.down1(x)

        x2_1_1 = self.down2_b1_sbr(x1)
        x2_1_2 = self.down2_b1_cbr(x2_1_1)
        x2_2_1 = self.down2_b2_sbr(x1)
        x2_2_2 = self.down2_b2_cbr(x2_2_1)
        x2_3 = torch.cat([x2_1_2, x2_2_2], dim=1)
        x2 = self.down2_cbr(x2_3)
        
        x3_1_1 = self.down3_b1_sbr(x2)
        x3_1_2 = self.down3_b1_cbr(x3_1_1)
        x3_2_1 = self.down3_b2_sbr(x2)
        x3_2_2 = self.down3_b2_cbr(x3_2_1)
        x3_3 = torch.cat([x3_1_2, x3_2_2], dim=1)
        x3 = self.down3_cbr(x3_3) # skip connection 1

        x4_1_1 = self.down4_b1_sbr(x3)
        x4_1_2 = self.down4_b1_cbr(x4_1_1)
        x4_2_1 = self.down4_b2_sbr(x3)
        x4_2_2 = self.down4_b2_cbr(x4_2_1)
        x4_3 = torch.cat([x4_1_2, x4_2_2], dim=1)
        x4 = self.down4_cbr(x4_3) # skip connection 2

        x5_1_1 = self.down5_b1_sbr(x4)
        x5_1_2 = self.down5_b1_cbr(x5_1_1)
        x5_2_1 = self.down5_b2_sbr(x4)
        x5_2_2 = self.down5_b2_cbr(x5_2_1)
        x5_3 = torch.cat([x5_1_2, x5_2_2], dim=1)
        x5 = self.down5_cbr(x5_3)

        ## up sampling
        x6_1_1 = self.up1_b1_dbr(x5)
        x6_1_2 = self.up1_b1_cbr(x6_1_1)
        x6_2_1 = self.up1_b2_dbr(x5)
        x6_2_2 = self.up1_b2_cbr(x6_2_1)
        x6 = torch.cat([x4, x6_1_2, x6_2_2], dim=1) # skip connection 2
        x6 = self.up1_cbr(x6)

        x7_1_1 = self.up2_b1_dbr(x6)
        x7_1_2 = self.up2_b1_cbr(x7_1_1)
        x7_2_1 = self.up2_b2_dbr(x6)
        x7_2_2 = self.up2_b2_cbr(x7_2_1)
        x7 = torch.cat([x3, x7_1_2, x7_2_2], dim=1) # skip connection 1
        x7 = self.up2_cbr(x7)

        x8_1_1 = self.up3_b1_dbr(x7)
        x8_1_2 = self.up3_b1_cbr(x8_1_1)
        x8_2_1 = self.up3_b2_dbr(x7)
        x8_2_2 = self.up3_b2_cbr(x8_2_1)
        x8 = torch.cat([x8_1_2, x8_2_2], dim=1)
        x8 = self.up3_cbr(x8)

        x9_1_1 = self.up4_b1_dbr(x8)
        x9_1_2 = self.up4_b1_cbr(x9_1_1)
        x9_2_1 = self.up4_b2_dbr(x8)
        x9_2_2 = self.up4_b2_cbr(x9_2_1)
        x9 = torch.cat([x9_1_2, x9_2_2], dim=1)
        x9 = self.up4_cbr(x9) # torch.Size([10, 64, 320, 256])

        logits = self.cbs(x9) # torch.Size([10, 9, 320, 256])
        
        return [logits]
        


def base_models(configs, **kwargs):
    print(configs)
    model = DetectionSubnetwork(configs, **kwargs)
    return model 