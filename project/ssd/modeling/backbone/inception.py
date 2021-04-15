import torch
from torch import nn
from torchvision.models import inception_v3 as inception


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

        inception_model = inception(pretrained = True, aux_logits=False)
        #print(inception_model)

        self.model = nn.Sequential(*(list(inception_model.children())[:-3]))
        print(self.model)

        ##freeze all
        #for param in self.model.parameters():
        #    param.requires_grad = False
        ## unfreeze some
        #for param in self.model[-3:].parameters():
        #    param.requires_grad = True



    def forward(self, x):
        out_features = []

        num_of_blocks = len(self.model)
        outputs_to_save = [9,13,15,16,17]

        for block in range(num_of_blocks):
            x = self.model[block](x)

            if block in outputs_to_save:
                out_features.append(x)

        return tuple(out_features)

