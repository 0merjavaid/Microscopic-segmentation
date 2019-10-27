import os
import ast
import json
import cv2
import time
import torch
import models
import argparse
import numpy as np
import torch.nn as nn
import utils.utils
from PIL import Image
from tqdm import tqdm
from utils import utils
from collections import defaultdict
from dataloader.loader import Inference_Loader
# Folder name/ Experiment name/ image category/ all images.jpg
from distutils.dir_util import copy_tree, remove_tree


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Cell instance segmentation using mask RCNN')

    parser.add_argument('--out_dir', default="./outputs/",
                        help='directory where output will be saved')
    parser.add_argument('--config_path', default="./config.json",
                        help='path of configuration file')
    parser.add_argument('--max_instances', type=int, default=350,
                        help='maximum number of instances for maskRCNN default is 500')

    args = parser.parse_args()

    return args


class Inference:

    def __init__(self, model_name, experiment_name, root_dir, weights_path, device, output_dir,
                 num_classes, classes, batch_size, backbone, num_instances):
        self.thres = 0.5
        self.device = device
        self.classes = classes
        self.out_dir = output_dir
        #self.out_dir = root_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.model = models.get_model(
            model_name, weights_path, num_classes, num_instances, backbone)
        self.model.to(device).eval()
        #self.root_dir = root_dir
        test_loader = Inference_Loader(root_dir)
        self.iterator = torch.utils.data.DataLoader(test_loader,
                                                    batch_size=batch_size,
                                                    shuffle=False, num_workers=4,
                                                    pin_memory=True)

    def infer(self):
        for sample in tqdm(self.iterator):
            directory, img_name, mask_img, unet_img, shape = sample
            with torch.no_grad():
                unet_img = unet_img.to(self.device)
                mask_img = mask_img.to(self.device)

                if self.model_name == "unet":
                    output = self.model(unet_img)
                    self.process_unet(output, img_name, directory,
                                      self.experiment_name, shape)
                elif self.model_name == "maskrcnn":
                    output = self.model(mask_img)
                    self.process_mask(output, img_name, directory,
                                      self.experiment_name, mask_img)
                elif self.model_name == "deeplab":
                    output = self.model(unet_img)["out"]
                    self.process_deeplab(
                        output, img_name, directory, self.experiment_name, shape)

    def process_deeplab(self, output, img_name, directory, experiment_name, shape):
        output = torch.max(output, 1)[1]
        output = output.cpu().detach().numpy()
        for i in range(len(output)):
            output_dir = os.path.join(
                self.out_dir, directory[i]+"_"+experiment_name)
            os.makedirs(output_dir, exist_ok=True)
            mask = output[i]
            mask = cv2.resize(mask.astype("uint8"),
                              (shape[i][1], shape[i][0]))
            cv2.imwrite(os.path.join(
                output_dir+"/" + img_name[i]+".png"), mask)

    def process_unet(self, output, img_name, directory, experiment_name, shape):
        output = output.cpu().detach().numpy()
        output = ((output > 0.5).astype(float)*255).astype("uint8")
        for i in range(len(output)):
            output_dir = os.path.join(
                self.out_dir, directory[i]+"_"+experiment_name)
            os.makedirs(output_dir, exist_ok=True)
            mask = output[i]
            mask = cv2.resize(mask, (shape[i][1], shape[i][0]))
            cv2.imwrite(os.path.join(
                output_dir+"/" + img_name[i]+".png"), mask)

    def process_mask(self, outputs, img_name, directory, experiment_name, images):
        images = images.cpu().detach().numpy()
        for i, output in enumerate(outputs):
            scores = output["scores"]
            bboxes = output["boxes"]
            mask = output["masks"].squeeze()
            classes = output["labels"]

            classes = classes[scores > self.thres]
            bboxes = bboxes[scores > self.thres]
            mask = mask[scores > self.thres]
            scores = scores[scores > self.thres]
            assert classes.shape[0] == bboxes.shape[0] == mask.shape[0]
            mask = mask.cpu().detach().numpy()
            img = np.transpose(images[i], [1, 2, 0])

            output_dir = os.path.join(
                self.out_dir, directory[i]+"_"+experiment_name)

            overlay, colored_mask, instances = self.visualize(
                img, mask, scores, bboxes, classes, self.thres)

            output_boxes = torch.cat(
                (classes.float().view(-1, 1), bboxes), 1).cpu().detach().numpy()
            current_out_dir = os.path.join(
                output_dir, "overlay")
            os.makedirs(current_out_dir, exist_ok=True)
            cv2.imwrite(current_out_dir+"/" +
                        img_name[i]+".png", overlay)
            current_out_dir = os.path.join(
                output_dir, "colored_mask")
            os.makedirs(current_out_dir, exist_ok=True)
            cv2.imwrite(current_out_dir+"/" +
                        img_name[i]+".png", colored_mask)

            for j, instance_class in enumerate(["combined"]+self.classes):
                current_out_dir = os.path.join(
                    output_dir, instance_class)
                os.makedirs(current_out_dir, exist_ok=True)
                cv2.imwrite(current_out_dir+"/" +
                            img_name[i]+".png", instances[j])
            current_out_dir = os.path.join(
                output_dir, "boxes")
            os.makedirs(current_out_dir, exist_ok=True)
            np.savetxt(current_out_dir+"/" +
                       img_name[i]+".txt", output_boxes.reshape(-1, 5))

    def visualize(self, image, mask, scores, boxes, classes,
                  threshold=0.5):

        final_mask = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        instances = np.zeros(
            (len(self.classes)+1, mask.shape[1], mask.shape[2])).astype("uint16")
        instance_ids = [1]*(len(self.classes)+1)

        for i, channel in enumerate(mask):
            cls = classes[i]
            channel[channel > threshold] = 1
            channel[channel <= threshold] = 0
            instances[0][channel == 1] = i+1
            instances[cls][channel == 1] = instance_ids[cls]
            instance_ids[cls] += 1
            final_mask[channel == 1] = np.random.randint(
                1, 255, size=3).tolist()
            final_mask[final_mask > 255] = 255

        for cls, box, score in zip(classes, boxes, scores):
            cls = cls.cpu().detach().item()
            score = score.cpu().detach().item()

            cv2.rectangle(final_mask, (box[0], box[1]),
                          (box[2], box[3]), (0, 255, 255), 2)
            cv2.putText(final_mask, self.classes[cls-1] + "    " + str(score)[
                        :4], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        final_mask = final_mask.astype("uint8")
        image = image.astype(float)*255
        image[final_mask != 0] -= 50
        image += final_mask/2

        image[image > 255] = 255
        return image.astype("uint8"), final_mask, instances


def parse_cfg(path="./config.json"):
    assert os.path.exists(path), "configuration file not found"
    cfg = dict()
    with open(path) as f:
        parser = json.loads(f.read())
    cfg["num_of_exp"] = parser["segmentation"]["num_of_exp"]
    cfg["root_dir"] = parser["root_dir"]
    cfg["experiments"] = parser["segmentation"]["experiments"]

    return cfg


def move_results(output_dir, root_dir):
    copy_tree(output_dir, root_dir)
    remove_tree(output_dir)


def main():
    args = get_argparser()
    cfg = parse_cfg(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else "cpu")
    number_of_experiments = int(cfg["num_of_exp"])
    root_dir = cfg["root_dir"]
    experiments = cfg["experiments"]
    assert number_of_experiments == len(cfg["experiments"])
    for i in range(number_of_experiments):
        experiment_name = experiments[i]["name"]
        model_name = experiments[i]["architecture"]
        num_classes = int(experiments[i]["num_classes"])
        classes = experiments[i]["class_names"]
        batch_size = int(experiments[i]["batch_size"])
        weights_path = experiments[i]["model_path"]
        try:
            backbone = experiments[i]["backbone"]
        except:
            backbone = None
        print ("Processing for ", model_name)

        inference = Inference(model_name, experiment_name, root_dir, weights_path,
                              device, root_dir, num_classes, classes,
                              batch_size, backbone, args.max_instances)

        inference.infer()
    # move_results(args.out_dir, root_dir)


if __name__ == "__main__":
    main()
