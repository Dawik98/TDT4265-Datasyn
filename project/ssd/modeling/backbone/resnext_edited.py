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
        self.model[0] = nn.Conv2d(image_channels, 64, kernel_size=3, stride=2, padding=2, bias=False)


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

