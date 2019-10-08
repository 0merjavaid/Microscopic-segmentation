import os
import ast
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
from dataloader.loader import Inference_Loader
# Folder name/ Experiment name/ image category/ all images.jpg


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Cell instance segmentation using mask RCNN')

    parser.add_argument('--out_dir', default="./outputs/",
                        help='directory where output will be saved')
    parser.add_argument('--config_path', default="./configuration.txt",
                        help='path of configuration file')
    parser.add_argument('--max_instances', type=int, default=350,
                        help='maximum number of instances for maskRCNN default is 500')

    args = parser.parse_args()

    return args


class Inference:

    def __init__(self, model_name, experiment_name, root_dir, weights_path, device, output_dir,
                 num_classes, classes, batch_size, num_instances):
        self.thres = 0.5
        self.device = device
        self.classes = classes
        self.out_dir = output_dir
        self.model_name = model_name
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.model = models.get_model(
            model_name, weights_path, num_classes, num_instances)
        self.model.to(device).eval()
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
            test_path = os.path.join(
                self.out_dir, directory[i], experiment_name, "deeplab", img_name[i]+".png")
            output_dir = os.path.join(
                self.out_dir, directory[i], experiment_name, "deeplab", img_name[i]+".png")
            os.makedirs(output_dir, exist_ok=True)
            mask = output[i]
            mask = cv2.resize(mask.astype("uint8"), (shape[i][1], shape[i][0]))
            cv2.imwrite(os.path.join(output_dir, "deeplab.png"), mask)

    def process_unet(self, output, img_name, directory, experiment_name, shape):
        output = output.cpu().detach().numpy()
        output = ((output > 0.5).astype(float)*255).astype("uint8")
        for i in range(len(output)):
            output_dir = os.path.join(self.out_dir, directory[i], img_name[i])
            os.makedirs(output_dir, exist_ok=True)
            mask = output[i]
            mask = cv2.resize(mask, (shape[i][1], shape[i][0]))
            cv2.imwrite(os.path.join(output_dir, "unet.png"), mask)

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

            overlay, colored_mask, instances = self.visualize(
                img, mask, scores, bboxes, classes, self.thres)

            output_boxes = torch.cat(
                (classes.float().view(-1, 1), bboxes), 1).cpu().detach().numpy()
            os.makedirs(os.path.join(self.out_dir, directory[i],
                                     img_name[i]), exist_ok=True)
            cv2.imwrite(os.path.join(self.out_dir, directory[i],
                                     img_name[i], "overlay.png"), overlay)
            cv2.imwrite(os.path.join(self.out_dir, directory[i],
                                     img_name[i], "colored_mask.png"), colored_mask)

            for j, instance_class in enumerate(["combined"]+self.classes):
                cv2.imwrite(os.path.join(
                    self.out_dir, directory[i], img_name[i], instance_class+".png"), instances[j])
            np.savetxt(os.path.join(
                self.out_dir, directory[i], img_name[i], "boxes.txt"), output_boxes.reshape(-1, 5))

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


def parse_cfg(path="./configuration.txt"):
    assert os.path.exists(path), "configuration file not found"

    with open(path, "r") as f:
        lines = f.readlines()
    start = False
    cfg = {'experiments': []}
    for line in lines:
        if len(line.strip()) == 0 or line[0] == "#":
            continue

        if not start and "=segmentation=" in line.strip().lower():
            start = True
            continue
        elif start and line.strip().lower()[0] == "=":
            break
        else:
            if line.strip().lower()[0] == "[":
                cfg['experiments'].append(
                    ast.literal_eval(line.strip().lower()))
            else:
                line = line.split("=")
                cfg[line[0].strip().lower()] = line[1].strip()
    return cfg


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
        experiment_name = experiments[i][0]
        model_name = experiments[i][1]
        num_classes = int(experiments[i][2])
        classes = experiments[i][3]
        batch_size = int(experiments[i][5])
        weights_path = experiments[i][4]

        print ("Processing for ", model_name)

        inference = Inference(model_name, experiment_name, root_dir, weights_path,
                              device, args.out_dir, num_classes, classes,
                              batch_size, args.max_instances)

        inference.infer()

if __name__ == "__main__":
    main()
