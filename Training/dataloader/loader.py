import os
import cv2
import glob
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt


class CellDataset(object):

    def __init__(self, root, transform, label_type, segmentation_type, classes, mode="Train"):
        self.root = root
        self.mode = mode
        self.lr = iaa.Sequential([iaa.Fliplr(1)])
        self.ud = iaa.Sequential([iaa.Flipud(1)])
        self.normalize = transforms.Normalize(mean=[
            0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
        self.prep = transforms.Compose([

            self.normalize
        ])
        self.classes = classes
        self.label_type = label_type
        self.transforms = transform
        self.segmentation_type = segmentation_type
        if label_type == "pairs":
            self.dataset = self.initialize_pairs()
        elif label_type == "json":
            self.dataset = self.initialize_jsons()

    def initialize_pairs(self):
        dataset = list()
        sets = sorted(os.listdir(self.root))

        sets = [i for i in sets if "set" in i]
        assert len(sets) > 1, "No Sets found inside the data directory"

        for a_set in sets:
            samples = list()
            set_path = os.path.join(self.root, a_set)
            tifs = glob.glob(os.path.join(set_path, "raw.tif"))
            pngs = glob.glob(os.path.join(set_path, "*png"))
            assert len(tifs) == 1, "Raw tif not found"
            label = tifs[0].replace("tif", "json")
            assert os.path.exists(label), "Label not found for " + file
            with open(label, "r") as f:
                label = json.load(f)
                shapes = label["shapes"]

                for shape in shapes:
                    cls = shape["label"]
                    points = shape["points"]

                    points = np.array(points).reshape(-1, 1, 2).astype(int)
                    samples.append([tifs[0], cls, points])
            dataset.append(samples)
        assert len(dataset) == len(sets)

        return dataset

    def initialize_jsons(self):
        dataset = list()
        files = glob.glob(os.path.join(self.root, "*tif"))
        assert len(files) > 0, "No Tif images found"
        for file in files:
            samples = list()
            label = file.replace("tif", "json")
            assert os.path.exists(label), "Label not found for " + file
            with open(label, "r") as f:
                label = json.load(f)
                shapes = label["shapes"]
                for shape in shapes:
                    cls = shape["label"]
                    points = shape["points"]
                    points = np.array(points).reshape(-1, 2).astype(int)
                    bbox = np.array(cv2.boundingRect(points))
                    samples.append([file, cls, points, bbox])
            dataset.append(samples)
        assert len(dataset) == len(files)
        return dataset

    def __getitem__(self, idx):
        # load images ad masks
        target = dict()
        image_list = self.dataset[idx]
        img_path = image_list[0][0]
        img = Image.open(img_path.replace("tif", "png")).convert("RGB")
        bin_mask = np.zeros((img.size[1], img.size[0]))
        if self.segmentation_type == "maskrcnn":
            masks = np.zeros((len(image_list), img.size[1], img.size[0]))
            all_points = list()
            boxes = list()
            classes = list()
            for i, (img_path, cls, points)in enumerate(image_list):

                masks[i] = cv2.drawContours(masks[i], [points], -1, 1, -1)
                pos = np.where(masks[i] == 1)

                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                bbox = [xmin, ymin, xmax, ymax]
                boxes.append(bbox)
                classes.append(self.classes[cls])

            # cv2.imwrite(img_path.replace(".tif", "_instances.jpg"), instances)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            classes = torch.as_tensor(classes, dtype=torch.int64)
            masks = torch.as_tensor(masks, dtype=torch.uint8)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((masks.shape[0],), dtype=torch.int64)
            assert len(boxes > 1) and len(boxes) < 350
            target["boxes"] = boxes
            target["labels"] = classes
            target["masks"] = masks
            target["image_id"] = torch.tensor([idx])
            target["area"] = area
            target["iscrowd"] = iscrowd
            if self.transforms is not None:
                img, target = self.transforms(img, target)
        else:
            all_points = [i[2] for i in image_list]
            bin_mask = cv2.drawContours(bin_mask, all_points, -1, 1, -1)
            img = np.array(img)
            if self.mode == "Train":
                if np.random.randint(0, 10) % 2 == 0:
                    img = self.lr.augment_image(np.array(img))
                    bin_mask = self.lr.augment_image(bin_mask)
                if np.random.randint(0, 10) % 2 == 0:
                    img = self.ud.augment_image(np.array(img))
                    bin_mask = self.ud.augment_image(bin_mask)

            if self.transforms is not None:

                img = self.transforms(Image.fromarray(img))
                img = self.prep(img)
                bin_mask = self.transforms(Image.fromarray(bin_mask))
                img = img.unsqueeze(0)
                target["image_id"] = torch.tensor([idx])
                target["boxes"] = torch.zeros(1, 4)
                target["labels"] = torch.zeros(1, )
                target["area"] = torch.zeros(1, 1)
                target["iscrowd"] = torch.zeros(1, 1)
                target["bin_mask"] = bin_mask.float()

        return img, target

    def __len__(self):
        return len(self.dataset)
