import os
from torchvision import transforms
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


def get_transform(model, train):
    transform = []
    if model.lower() == "maskrcnn":
        transform.append(T.ToTensor())
        if train:
            transform.append(T.RandomHorizontalFlip(0.5))
            transform.append(T.RandomVerticalFlip(0.5))
    else:
        preprocess = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[
            #     0.485, 0.456, 0.406], std=[
            #     0.229, 0.224, 0.225])
        ])

        return preprocess
    return T.Compose(transform)
