import torch
from torch import nn
from torchvision.models import resnet152 as resnet


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

        resnet_model = resnet(pretrained = True)
        model = [l for l in resnet_model.children()][:-2]
        print(model)

        # freeze layers
        freeze_layers = [0,1,2]
        for layer in freeze_layers:
            for param in model[layer].parameters():
                param.requires_grad = False

        # unfreeze layers
        for param in model[3].parameters():
            param.requires_grad = True

        self.layer1 = model[0]
        self.layer2 = model[1]
        self.layer3 = model[2]
        self.layer4 = model[3]

        self.last_block = nn.Sequential(
            nn.Conv2d(64, 2048, 3, 1, 0)
        )

    def forward(self, x):
        out_features = []

        x = self.layer1(x)
        out_features.append(x)
        x = self.layer2(x)
        out_features.append(x)
        x = self.layer3(x)
        out_features.append(x)
        x = self.layer4(x)
        out_features.append(x)
        #x = self.last_block(x)
        #out_features.append(x)

        for i in out_features:
            print(i.size())

        return tuple(out_features)

