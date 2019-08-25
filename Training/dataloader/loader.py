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
                    if len(points) > 1:
                        points = [np.array(cnt).reshape(-1, 1, 2).astype(int)
                                  for cnt in points]
                    else:
                        points = [
                            np.array(points).reshape(-1, 1, 2).astype(int)]
                    # bbox = np.array(cv2.boundingRect(points))
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
        img1 = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(img_path.replace("tif", "png")).convert("RGB")
        masks = np.zeros((len(image_list), img1.shape[0], img1.shape[1]))
        bin_mask = np.zeros((img1.shape[0], img1.shape[1]))
        all_points = list()
        boxes = list()
        classes = list()
        instances = np.zeros_like(bin_mask)
        for i, (img_path, cls, points)in enumerate(image_list):

            masks[i] = cv2.drawContours(masks[i], points, -1, 1, -1)
            pos = np.where(masks[i] == 1)

            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            bbox = [xmin, ymin, xmax, ymax]
            # print (len(points))
            instances = cv2.drawContours(instances, points, -1, i, -1)
            all_points.append(points)
            # bbox = box.copy()
            # bbox[2] = bbox[0] + bbox[2]
            # bbox[3] = bbox[1] + bbox[3]
            boxes.append(bbox)
            classes.append(self.classes[cls])
        cv2.imwrite(img_path.replace(".tif", "_instances.jpg"), instances)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        # bin_mask = cv2.drawContours(bin_mask, all_points, -1, 1, -1)
        bin_mask = torch.as_tensor(bin_mask, dtype=torch.uint8)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((masks.shape[0],), dtype=torch.int64)
        assert len(boxes > 1) and len(boxes) < 350
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
