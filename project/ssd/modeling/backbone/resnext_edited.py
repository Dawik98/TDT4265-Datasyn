import torch
from torch import nn
from torchvision.models import resnext50_32x4d as resnext
import copy
#from torchvision.models import vgg11_bn


class Model(torch.nn.Module):
    """
    This is  backbone for SSD.
    """

    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        resnext_model = resnext(pretrained = True)
        #print(resnext_model)

        #vgg = vgg11_bn(pretrained=True)
        #vgg.features[0].stride = (2,2)
        #self.block0 = nn.Sequential(*(list(vgg.features.children())[:4]))

        self.model = nn.Sequential(*(list(resnext_model.children())[:-2]))

        ## block 4
        #self.model[4][0].conv2.dilation = (2, 2)
        #self.model[4][0].conv2.padding = (2, 2)
        #self.model[4][1].conv2.dilation = (2, 2)
        #self.model[4][1].conv2.padding = (2, 2)
        #self.model[4][2].conv2.dilation = (2, 2)
        #self.model[4][2].conv2.padding = (2, 2)

        # block 5
        self.model[5][0].conv2.dilation = (2, 2)
        self.model[5][0].conv2.padding = (2, 2)
        self.model[5][1].conv2.dilation = (2, 2)
        self.model[5][1].conv2.padding = (2, 2)
        self.model[5][2].conv2.dilation = (2, 2)
        self.model[5][2].conv2.padding = (2, 2)
        self.model[5][3].conv2.dilation = (2, 2)
        self.model[5][3].conv2.padding = (2, 2)

        # block 6
        self.model[6][0].conv2.dilation = (2, 2)
        self.model[6][0].conv2.padding = (2, 2)
        self.model[6][1].conv2.dilation = (2, 2)
        self.model[6][1].conv2.padding = (2, 2)
        self.model[6][2].conv2.dilation = (2, 2)
        self.model[6][2].conv2.padding = (2, 2)
        self.model[6][3].conv2.dilation = (2, 2)
        self.model[6][3].conv2.padding = (2, 2)
        self.model[6][4].conv2.dilation = (2, 2)
        self.model[6][4].conv2.padding = (2, 2)
        self.model[6][5].conv2.dilation = (2, 2)
        self.model[6][5].conv2.padding = (2, 2)

        # block 7
        self.model[7][0].conv2.dilation = (2, 2)
        self.model[7][0].conv2.padding = (2, 2)
        self.model[7][1].conv2.dilation = (2, 2)
        self.model[7][1].conv2.padding = (2, 2)
        self.model[7][2].conv2.dilation = (2, 2)
        self.model[7][2].conv2.padding = (2, 2)



        print(self.model)

        ##freeze all
        #for param in self.model.parameters():
        #    param.requires_grad = False
        ## unfreeze some
        #for param in self.model[-3:].parameters():
        #    param.requires_grad = True




    def forward(self, x):
        out_features = []

        x = self.model[:6](x)
        out_features.append(x)
        x = self.model[6](x)
        out_features.append(x)
        x = self.model[7](x)
        out_features.append(x)

        return tuple(out_features)

