import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from ssd.data import samplers
from ssd.data.datasets import build_dataset
from ssd.data.transforms import build_transforms, build_target_transform
from ssd.container import Container


class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])

        if self.is_train:
            list_targets = transposed_batch[1]
            targets = Container(
                {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
            )
        else:
            targets = None
        return images, targets, img_ids

def analyze_dataset(dataset):
    import numpy as np
    from matplotlib import pyplot as plt

    n = 0
    n_classes = [0,0,0,0,0]
    mean = 0.0
    std = 0.0

    img_size = max(360,645)
    x_corners = np.zeros(img_size, dtype=int)
    y_corners = np.zeros(img_size, dtype=int)

    box_widths = np.zeros(img_size, dtype=int)
    box_heights = np.zeros(img_size, dtype=int)

    aspect_ratios = np.zeros(12, dtype=int)

    for img in dataset:
        mean += torch.mean(img[0], (1,2))
        std += torch.std(img[0], (1,2))

        #count labels
        for l in img[1]['labels']:
            n_classes[l] += 1

        #box position distribution
        for box in img[1]['boxes']:
            x1 = int(box[0]*640)
            x2 = int(box[2]*640)
            y1 = int(box[1]*360)
            y2 = int(box[3]*360)

            box_w = abs(x2-x1)
            box_h = abs(y2-y1)

            if box_h == 0 or box_w == 0:
                continue

            ratio = box_w / box_h
            if ratio < 1:
                ratio = 1/ratio
            if abs(ratio) > 10:
                ratio = 10.0
            if ratio < 0:
                ratio = -ratio
            aspect_ratios[int(ratio)] += 1

            x_corners[x1] += 1
            x_corners[x2] += 1
            y_corners[y1] += 1
            y_corners[y2] += 1

            box_widths[box_w] += 1
            box_heights[box_h] += 1

        n += 1


    x_axis = np.linspace(0,img_size,img_size)

    plt.plot(x_axis, x_corners, label='x_coords')
    plt.plot(x_axis, y_corners, label='y_coords')
    plt.show()

    plt.plot(x_axis, box_widths, label='box_widths')
    plt.plot(x_axis, box_heights, label='box_heights')
    plt.show()

    plt.plot([1,2,3,4,5,6,7,8,9,10,11,12], aspect_ratios, label='box_widths')
    plt.show()

    mean /= n
    std /= n

    print("Dataset summary:")
    print("Data mean: {}".format(mean))
    print("Data std: {}".format(std))
    print("Classes: {}".format(n_classes))


def make_data_loader(cfg, is_train=True, max_iter=None, start_iter=0):
    train_transform = build_transforms(cfg, is_train=is_train)
    target_transform = build_target_transform(cfg) if is_train else None
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = build_dataset(
        cfg.DATASET_DIR,
        dataset_list, transform=train_transform,
        target_transform=target_transform, is_train=is_train)

    #analyze_dataset(datasets[0]) #uncomment to get info about dataset (remember to remove transforms)

    shuffle = is_train

    data_loaders = []

    for dataset in datasets:
        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.sampler.SequentialSampler(dataset)

        batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=is_train)
        if max_iter is not None:
            batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=max_iter, start_iter=start_iter)

        data_loader = DataLoader(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                                 pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(is_train))
        data_loaders.append(data_loader)

    if is_train:
        # during training, a single (possibly concatenated) data_loader is returned
        assert len(data_loaders) == 1
        return data_loaders[0]
    return data_loaders
