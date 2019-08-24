import os
import cv2
import glob
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


class CellDataset(object):

    def __init__(self, root, transforms, label_type, segmentation_type, classes):
        self.root = root
        self.classes = classes
        self.label_type = label_type
        self.transforms = transforms
        self.segmentation_type = segmentation_type
        if label_type == "pairs":
            self.dataset = self.initialize_pairs()
        elif label_type == "json":
            self.dataset = self.initialize_jsons()
        # load all image files, sorting them to
        # ensure that they are aligned
        # self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        # self.masks = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def initialize_pairs(self):
        dataset = list()
        sets = os.listdir(self.root)
        sets = [i for i in sets if "set" in i]
        assert len(sets) > 1, "No Sets found inside the data directory"

        for a_set in sets:
            set_path = os.path.join(self.root, a_set)
            tifs = glob.glob(os.path.join(set_path, "raw.tif"))
            pngs = glob.glob(os.path.join(set_path, "*png"))
            assert len(tifs) == 1, "Raw tif not found"
            masks = list()
            for key in self.classes.keys():
                class_id = self.classes[key]
                class_name = key + ".png"
                class_label_path = os.path.join(set_path, class_name)
                assert class_label_path in pngs, class_name+" Not Found"
                masks.append((class_label_path, class_id))

            one_class_seg = glob.glob(os.path.join(set_path, "feature_1.png"))
            assert len(one_class_seg) == 1, "Feature 1 image not found"

            masks.append((one_class_seg[0], -1))
            img_path = tifs
            masks.append((img_path[0], 0))
            dataset.append(masks)
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
        img = cv2.imread(img_path)
        masks = np.zeros((len(image_list), img.shape[0], img.shape[1]))
        bin_mask = np.zeros((img.shape[0], img.shape[1]))
        all_points = list()
        boxes = list()
        classes = list()
        for i, (img_path, cls, points, bbox)in enumerate(image_list):
            masks[i] = cv2.fillPoly(masks[i], np.int_([points]), 1)
            all_points.append(points)

            bbox[2] = bbox[0] + bbox[2]
            bbox[3] = bbox[1] + bbox[3]
            boxes.append(bbox)
            classes.append(self.classes[cls])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        bin_mask = cv2.drawContours(bin_mask, all_points, -1, 1, -1)
        bin_mask = torch.as_tensor(bin_mask, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((masks.shape[0],), dtype=torch.int64)

        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["binmask"] = bin_mask

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.dataset)
