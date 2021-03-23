

# VGG16 ispired 
class VGG16:
        self.block1 = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.MaxPool2d(f_pool, s_pool),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # size out = 150x150x32
            # ------------------------------------------------------
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(f_pool, s_pool),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # size out = 75x75x64
            # ------------------------------------------------------
        )
        self.block3 = nn.Sequential(
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
        self.block4 = nn.Sequential(
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
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # size out = 9x9x256

            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # size out = 5x5x256

            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # size out = 3x3x256
        )
            # --------------------------------------------------------
        self.block6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Conv2d(512, self.output_channels[5], 3, 1, 0),
            # --------------------------------------------------------
        )



class org_modified:
        self.block1 = nn.Sequential(
            nn.Conv2d(image_channels, 32, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.MaxPool2d(f_pool, s_pool),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(f_pool, s_pool),
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, self.output_channels[0], 3, 2, 1),
            # ------------------------------------------------------
        )
        self.block2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(self.output_channels[0]),

            nn.Conv2d(self.output_channels[0], 128, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, 5, 1, 2),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, self.output_channels[1], 3, 2, 1),
            # ------------------------------------------------------
        )
        self.block3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(self.output_channels[1]),

            nn.Conv2d(self.output_channels[1], 256, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(256),


            #nn.Conv2d(256, 256, 5, 1, 2),
            #nn.Dropout(0.2),
            #nn.ReLU(),
            #nn.BatchNorm2d(256),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, self.output_channels[2], 3, 2, 1),
            # ------------------------------------------------------
        )
        self.block4 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(self.output_channels[2]),

            nn.Conv2d(self.output_channels[2], 128, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            #nn.Conv2d(256, 256, 3, 1, 1),
            #nn.Dropout(0.2),
            #nn.ReLU(),
            #nn.BatchNorm2d(256),

            nn.Conv2d(128, self.output_channels[3], 3, 2, 1),
            # ------------------------------------------------------
        )
        self.block5 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(self.output_channels[3]),

            nn.Conv2d(self.output_channels[3], 256, 3, 1, 1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, self.output_channels[4], 3, 2, 1),
            nn.Dropout(0.3),
        )
            # --------------------------------------------------------
        self.block6 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm2d(self.output_channels[4]),

            nn.Conv2d(self.output_channels[4], 512, 3, 1, 1),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Conv2d(512, self.output_channels[5], 3, 1, 0),
            # --------------------------------------------------------
        )