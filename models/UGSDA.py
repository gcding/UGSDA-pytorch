import torch
import torch.nn as nn
import torch.nn.functional as F


class UGSDA(nn.Module):
    def __init__(self, model_name):
        super(UGSDA, self).__init__()
        if model_name == "UGSDA":
            from .SASNet_own import SASNet_own as ccnet
        else:
            raise ValueError('Network cannot be recognized. Please define your own Network.')

        self.CCN = ccnet()

        print("Model {} init success".format(model_name))
    
    def forward(self, img):
        pass

    def test_forward(self, img):                               
        density_map = self.CCN(img)
        return density_map

