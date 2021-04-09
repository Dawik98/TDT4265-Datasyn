import torch
from torch import nn
from torchvision.models import densenet121 as densenet


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

        densenet_model = densenet(pretrained = True)

        self.model = nn.Sequential(*(list(densenet_model.children())[:-1][0]))
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
        x = self.model[6:8](x)
        out_features.append(x)
        x = self.model[8:10](x)
        out_features.append(x)
        x = self.model[10:12](x)
        out_features.append(x)

        return tuple(out_features)

