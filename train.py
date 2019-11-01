import os
import cv2
import time
import torch
import models
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from utils import utils
from trainer import trainer
from dataloader import loader


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Cell instance segmentation using mask RCNN')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='Number of epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--num_classes', type=int, default=2, required=True,
                        help='Number of classes in case of maskRCNN, \
                        for example if you only have cell and\
                         background then num_classes 2')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate, deeplab 5x10-4, UNET , MaskRCNN .')
    parser.add_argument('--labels_type', default='pairs',
                        help='what is the type of labels? "pairs" or "json"', required=True)
    parser.add_argument('--root_dir', default='datasets/images',
                        help='directory that contains images and labels folders', required=True)
    parser.add_argument('--weight_path', default=None,
                        help='path to model weight file to be loaded')
    parser.add_argument('--checkpoint_dir', default='checkpoints', metavar='LR',
                        help='directory path to store checkpoints')
    parser.add_argument('--model', default='maskRCNN',
                        help='which model to use for training? MaskRCNN or UNET', required=True)
    parser.add_argument('--max_instances', type=int, default=350,
                        help='maximum number of instances for maskRCNN default is 500')
    parser.add_argument('--config_path', default='config.txt',
                        help='a File containing the names of classes')
    parser.add_argument('--maskrcnn_backbone', default='resnet101',
                        help='resnet101 or resnet50 for maskrcnn backbone')

    args = parser.parse_args()
    assert args.labels_type in [
        "pairs", "json"]

    if torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False

    return args


def main():
    args = get_argparser()

    # our dataset has two class
    classes = utils.parse_config(args.config_path)
    print (len(classes), args.num_classes, classes)
    assert len(
        classes)+1 == args.num_classes, "Number of classes\
    in config and argument is not same"
    # use our dataset and defined transformations

    dataset = loader.CellDataset(
        args.root_dir, utils.get_transform(args.model, train=True),
        args.labels_type, args.model, classes)
    dataset_test = loader.CellDataset(
        args.root_dir, utils.get_transform(
            args.model, train=False), args.labels_type,
        args.model, classes, mode="Test")

    indices = torch.arange(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:int(len(indices)*0.9)])
    dataset_test = torch.utils.data.Subset(
        dataset_test, indices[int(len(indices)*0.9):])
    print ("Images in Test set", len(dataset_test),
           "Images in Train set ", len(dataset))
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    model = models.get_model(
        args.model, args.weight_path, args.num_classes, args.max_instances, args.maskrcnn_backbone)

    if args.cuda:
        device = "cuda:0"
        model.to(device)
    else:
        device = "cpu"

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
    #                             momentum=0.9, weight_decay=0.0005)
    print ("\n\nStarting Training of ", args.model, "\n\n")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model_trainer = trainer.TrainModel(model, optimizer, args.model, device)
    for epoch in range(args.epochs):
        model_trainer.train(epoch, data_loader, data_loader_test)

    print("That's it!")


if __name__ == '__main__':
    main()
