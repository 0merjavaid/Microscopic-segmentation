from dataloader import loader
import models
import numpy as np
import cv2
from PIL import Image
# import trainer
# import utils
import argparse
import torch
from tqdm import tqdm
from utils import utils
import os
from mask_utils.engine import train_one_epoch, evaluate


def get_argparser():
    parser = argparse.ArgumentParser(
        description='Cell instance segmentation using mask RCNN')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--num_classes', type=int, default=2, required=True,
                        help='Number of classes in case of maskRCNN, \
                        for example if you only have cell and\
                         background then num_classes 2')
    parser.add_argument('--lr', type=float, default=0.005, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs')
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
    parser.add_argument('--infer', type=int, default=0,
                        help='--infer 1 if you want to se results')
    parser.add_argument('--out_dir', default="./outputs/",
                        help='directory where output will be saved')

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
    # use our dataset and defined transformations
    dataset = loader.CellDataset(
        args.root_dir, utils.get_transform(train=True), args.labels_type, args.model, classes)
    dataset_test = loader.CellDataset(
        args.root_dir, utils.get_transform(train=False), args.labels_type,
        args.model, classes)
    assert len(
        classes)+1 == args.num_classes, "Number of classes\
        in config and argument is not same"
    indices = torch.arange(len(dataset)).tolist()
    print (len(indices))
    dataset = torch.utils.data.Subset(dataset, indices[:45])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-5:])
    print (len(dataset), len(dataset_test))
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=2, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = models.get_model(
        args.model, args.max_instances, args.weight_path, args.num_classes+1)
    if args.cuda:
        model.cuda()
        device = "cuda"
    else:
        device = "cpu"

    if args.infer:
        infer(model, dataset_test, args.out_dir)
        return
    # move model to the right device

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.05)

    # let's train it for 10 epochs
    num_epochs = 10
#     model.load_state_dict(torch.load("checkpoints/8_epoch.pt"))
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=5)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if (epoch) % 5 == 0:
            evaluate(model, data_loader_test, device=device)
            path = "checkpoints/"+str(epoch)+"_epoch.pt"
            torch.save(model.state_dict(), path)
    return model, data_loader_test, data_loader
    print("That's it!")


def visualize(raw, image, scores, boxes, classes, threshold=0.6):
    final_mask = np.zeros((image.shape[1], image.shape[2], 3))
    instances = np.zeros((image.shape[1], image.shape[2])).astype("uint16")
    for i, channel in enumerate(image):
        channel[channel > threshold] = 1
        channel[channel <= threshold] = 0
        instances[channel == 1] = i
        final_mask[channel == 1] = np.random.randint(1, 255, size=3).tolist()
        final_mask[final_mask > 255] = 255
    for cls, box, score in zip(classes, boxes, scores):
        cls = cls.cpu().detach().item()
        score = score.cpu().detach().item()
        cv2.rectangle(final_mask, (box[0], box[1]),
                      (box[2], box[3]), (0, 0, 255), 2)
        cv2.putText(final_mask, str(cls)+"    "+str(score)[:4], (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

    final_mask = final_mask.astype("uint8")
    raw = raw.astype(float)*255
    raw[final_mask != 0] -= 50
    raw += final_mask/2

    raw[raw > 255] = 255
    # plt.figure(figsize=(20, 10))
    # plt.imshow(final_mask)
    # plt.show()
    print ("Final Shape:", final_mask.shape)
    return raw.astype("uint8"), final_mask, instances


def infer(model, dataloader, out_dirs, thres=0.5):
    if not os.path.exists(out_dirs):
        os.mkdir(out_dir)
    model.eval()
    i = 0
    for image, target in dataloader:
        image = list(img.to("cuda") for img in [image])
        target_masks = target["masks"].cpu().detach().numpy()
        target_boxes = target["boxes"]
        # lab = visualize(target_masks,
        #                 [1]*target_masks.shape[0], target_boxes, thres)

        outputs = model(image)
        scores = outputs[0]["scores"]
        bboxes = outputs[0]["boxes"]
        mask = outputs[0]["masks"].squeeze()
        classes = outputs[0]["labels"]

        classes = classes[scores > thres]
        bboxes = bboxes[scores > thres]
        mask = mask[scores > thres]
        scores = scores[scores > thres]
        assert classes.shape[0] == bboxes.shape[0] == mask.shape[0]
        mask = mask.cpu().detach().numpy()

    #     print(len(image))
        img = image[0].cpu().detach().numpy()
        img = np.transpose(img, [1, 2, 0])
        overlay, colored_mask, instances = visualize(
            img, mask, scores, bboxes, classes, thres)
        cv2.imwrite(os.path.join(out_dirs, str(i)+"_raw.jpg"),
                    (img*255).astype("uint8"))
        cv2.imwrite(os.path.join(out_dirs, str(i)+"_instances.png"), instances)
        cv2.imwrite(os.path.join(out_dirs, str(i)+"_overlay.jpg"), overlay)
        cv2.imwrite(os.path.join(out_dirs, str(
            i)+"_color_mask.jpg"), colored_mask)
        i += 1

if __name__ == '__main__':
    main()
