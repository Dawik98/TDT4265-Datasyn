import torch
from torch import nn
from torchvision.models import resnext50_32x4d as resnext


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

        self.model = nn.Sequential(*(list(resnext_model.children())[:-2]))
        self.model[0] = nn.Conv2d(image_channels, 64, kernel_size=3, stride=2, padding=2, dilation=2, bias=False)


        print(self.model)

        ##freeze all
        #for param in self.model.parameters():
        #    param.requires_grad = False
        ## unfreeze some
        #for param in self.model[-3:].parameters():
        #    param.requires_grad = True

    def get_downsample(self, channels_in, channels_out):
        downsample = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
            #nn.BatchNorm2d(channels_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        return downsample


    def forward(self, x):
        out_features = []

        x = self.model[:6](x)
        out_features.append(x)
        x = self.model[6](x)
        out_features.append(x)
        x = self.model[7](x)
        out_features.append(x)

        return tuple(out_features)

