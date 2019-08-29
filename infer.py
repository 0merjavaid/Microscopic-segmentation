import os
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


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Cell instance segmentation using mask RCNN')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--num_classes', type=int, default=4, required=True,
                        help='Number of classes in case of maskRCNN, \
                        for example if you only have cell and\
                         background then num_classes 2')
    parser.add_argument('--root_dir', default='datasets/images',
                        help='directory that contains images and labels\
                         folders', required=True)
    parser.add_argument('--maskrcnn_weight', default=None, required=True,
                        help='path to model weight file to be loaded')
    parser.add_argument('--unet_weight', default=None,
                        help='path to model weight file to be loaded',
                        required=True)
    parser.add_argument('--config_path', default='config.txt',
                        help='a File containing the names of classes')
    parser.add_argument('--out_dir', default="./outputs/",
                        help='directory where output will be saved')
    parser.add_argument('--max_instances', type=int, default=350,
                        help='maximum number of instances for maskRCNN default is 500')

    args = parser.parse_args()

    return args


class Inference:

    def __init__(self, root_dir, unet_weight, maskrcnn_weight, device, config, output_dir,
                 num_classes, batch_size, num_instances):
        assert unet_weight is not None and maskrcnn_weight is not None
        self.device = device
        self.thres = 0.5
        self.out_dir = output_dir
        self.batch_size = batch_size
        self.unet = models.get_model("unet", unet_weight)
        self.config = utils.parse_config(config)
        self.maskrcnn = models.get_model("maskrcnn", maskrcnn_weight,
                                         num_classes, num_instances)
        self.unet.to(device).eval()
        self.maskrcnn.to(device).eval()
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

                unet_out = self.unet(unet_img)
                mask_out = self.maskrcnn(mask_img)

                self.process_unet(unet_out, img_name, directory, shape)
                self.process_mask(mask_out, img_name, directory, mask_img)

    def process_unet(self, output, img_name, directory, shape):
        output = output.cpu().detach().numpy()
        output = ((output > 0.5).astype(float)*255).astype("uint8")
        for i in range(self.batch_size):
            output_dir = os.path.join(self.out_dir, directory[i], img_name[i])
            os.makedirs(output_dir, exist_ok=True)
            mask = output[i]
            mask = cv2.resize(mask, (shape[i][1], shape[i][0]))
            cv2.imwrite(os.path.join(output_dir, "unet.png"), mask)

    def process_mask(self, outputs, img_name, directory, images):
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
                img, mask, scores, bboxes, classes, self.config, self.thres)

            output_boxes = torch.cat(
                (classes.float().view(-1, 1), bboxes), 1).cpu().detach().numpy()
            os.makedirs(os.path.join(self.out_dir, directory[i],
                                     img_name[i]), exist_ok=True)
            cv2.imwrite(os.path.join(self.out_dir, directory[i],
                                     img_name[i], "overlay.png"), overlay)
            cv2.imwrite(os.path.join(self.out_dir, directory[i],
                                     img_name[i], "colored_mask.png"), colored_mask)

            for j, instance_class in enumerate(["combined"]+list(self.config.keys())):
                cv2.imwrite(os.path.join(
                    self.out_dir, directory[i], img_name[i], instance_class+".png"), instances[j])
            np.savetxt(os.path.join(
                self.out_dir, directory[i], img_name[i], "boxes.txt"), output_boxes.reshape(-1, 5))

    def visualize(self, image, mask, scores, boxes, classes, mapping,
                  threshold=0.5):
        final_mask = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        instances = np.zeros(
            (len(mapping)+1, mask.shape[1], mask.shape[2])).astype("uint16")
        instance_ids = [1]*len(mapping)
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
            cv2.putText(final_mask, str(list(mapping.keys())[cls-1])+"    " +
                        str(score)[:4], (box[0], box[1]
                                         ), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

        final_mask = final_mask.astype("uint8")
        image = image.astype(float)*255
        image[final_mask != 0] -= 50
        image += final_mask/2

        image[image > 255] = 255
        return image.astype("uint8"), final_mask, instances


def main():
    args = get_argparser()
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")
    inference = Inference(args.root_dir, args.unet_weight, args.maskrcnn_weight,
                          device, args.config_path, args.out_dir, args.num_classes,
                          args.batch_size, args.max_instances)

    inference.infer()

if __name__ == "__main__":
    main()
