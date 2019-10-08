import os
import cv2
import glob
import json
import torch
import random
import numpy as np
from skimage.transform import rotate
from PIL import Image
from torch.utils import data
import matplotlib.pyplot as plt
from torchvision import transforms


class CellDataset(object):

    def __init__(self, root, transform, label_type, segmentation_type, classes, mode="Train"):
        self.root = root
        self.mode = mode
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

    def augment(self, img, mask):
        if random.randint(0, 10) % 2 == 0:
            img = np.flipud(img)
            mask = np.flipud(mask)

        if random.randint(0, 10) % 2 == 0:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        if random.randint(0, 100) % 2 == 0:
            angle = [90, 180, 270]
            angle = angle[random.randint(0, 2)]
            img = (rotate(img, angle)*255).astype("uint8")
            mask = rotate(mask, angle)

        return img, mask

    def __getitem__(self, idx):
        # load images ad masks
        target = dict()
        image_list = self.dataset[idx]
        img_path = image_list[0][0]

        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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

                img, bin_mask = self.augment(img, bin_mask)

            if self.transforms is not None:

                img = self.transforms(Image.fromarray(img))
                img = self.prep(img)
                bin_mask = self.transforms(
                    Image.fromarray(bin_mask.squeeze().astype(float)))
                img = img.unsqueeze(0)
                target["bin_mask"] = bin_mask.float()

        return img, target

    def __len__(self):
        return len(self.dataset)


class Inference_Loader(data.Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.dataset = self.initialize()
        normalize = transforms.Normalize(mean=[
            0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
        self.transform_maskRCNN = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.transform_unet = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            # normalize,
        ])

    def initialize(self):
        dataset = list()
        dirs = os.listdir(self.root_dir)
        assert len(dirs) > 0, "No directory found containing images"
        # print (dirs)
        for dir in dirs:
            img_files = [os.path.join(self.root_dir, dir, i)
                         for i in ['*.tif', '*.png', "*.jpg"]]
            files = list()
            for e in img_files:
                files.extend(glob.glob(e))

            assert len(files) > 0, "No png, tif or jpg files found\
                in folder " + dir
            for file in files:
                dataset.append([dir, file])

        assert len(dataset) > 0, "No files found"
        # print (dataset)
        return dataset

    def __getitem__(self, idx):
        directory, path = self.dataset[idx]
        # print (path)
        img_name = path.split("/")[-1].split(".")[0]
        image = cv2.imread(path)
        shape = image.shape
        assert image.ndim > 1, "unable to load image" + img_name
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        mask_rcnn_image = self.transform_maskRCNN(image)
        unet_image = self.transform_unet(image)

        return directory, img_name, mask_rcnn_image, unet_image, np.array([shape[0], shape[1]])

    def __len__(self):
        return len(self.dataset)
