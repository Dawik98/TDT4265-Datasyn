import torch
from torch import nn


class Model(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        # parameters for maxpool layers
        f_pool = 2
        s_pool = 2


        self.block1 = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.MaxPool2d(f_pool, s_pool),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # size out = 150x150x32

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(f_pool, s_pool),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # size out = 75x75x64

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # size out = 38x38x128
            # ------------------------------------------------------
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # size out = 19x19x256
            # ------------------------------------------------------
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # size out = 10x10x256
            # ------------------------------------------------------
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # size out = 5x5x256
            # ------------------------------------------------------
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            # size out = 3x3x256
        )
            # --------------------------------------------------------
        self.block6 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, self.output_channels[5], 3, 1, 0),
            # --------------------------------------------------------
        )


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """

        out_features = []

        x = self.block1(x)
        out_features.append(x)
        x = self.block2(x)
        out_features.append(x)
        x = self.block3(x)
        out_features.append(x)
        x = self.block4(x)
        out_features.append(x)
        x = self.block5(x)
        out_features.append(x)
        x = self.block6(x)
        out_features.append(x)

        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            out_channel = self.output_channels[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

