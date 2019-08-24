import os
import mask_utils.transforms as T


def parse_config(path):
    class_to_ids = dict()
    assert os.path.exists(path)
    with open(path, "r") as f:
        lines = f.readlines()

    for id, line in enumerate(lines):
        line = line.strip().lower()
        class_to_ids[line] = id+1
    assert len(class_to_ids) >= 1
    return class_to_ids


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
