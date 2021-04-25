from ssd.modeling.box_head.prior_box import PriorBox
from .target_transform import SSDTargetTransform
from .transforms import *
from torchvision import transforms as torch_trans


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            #RandomRotate(3, 0.4),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            ToTensor(),
            RandomMirror(),
            RandomSampleCrop(),
            #RandomEffect(),
            #RandomPixelErasing(),
            #RandomEraseTop()
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_SIZE),
            SubtractMeans(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD),
            ToTensor(),
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = SSDTargetTransform(PriorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.THRESHOLD)
    return transform
