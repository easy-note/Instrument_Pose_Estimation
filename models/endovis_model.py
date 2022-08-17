
import torch
import torch.nn as nn

from .detection_subnetwork import DetectionSubnetwork
from .regression_subnetwork import RegressionSubnetwork

class Endovis(nn.Module):
    def __init__(self, configs, **kwargs):
        super(Endovis, self).__init__()

        self.detect_model = DetectionSubnetwork(configs, **kwargs)
        self.regression_model = RegressionSubnetwork(configs, **kwargs)

        self.detect_model.apply(self.weight_init_xavier_uniform)
        self.regression_model.apply(self.weight_init_xavier_uniform)
    def forward(self, x):
        # x.shape torch.Size([32, 3, 320, 256])
        detect_output = self.detect_model(x) # detect_output[0].shape torch.Size([32, 9, 320, 256])

        x = torch.cat([x, detect_output], dim=1) # torch.Size([32, 12, 320, 256])
        regression_output = self.regression_model(x)

        return [detect_output, regression_output]


    def weight_init_xavier_uniform(self, submodule):
        if isinstance(submodule, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(submodule.weight)
            # submodule.bias.fill_(0.01)
        elif isinstance(submodule, torch.nn.BatchNorm2d):
            # submodule.weight.data.fill_(1.0)
            # submodule.bias.data.zero_()
            nn.init.constant_(submodule.weight, 1)
            nn.init.constant_(submodule.bias, 0)

def endovis_models(configs, **kwargs):
    print(configs, '\n')
    model = Endovis(configs, **kwargs)
    return model  