from dataloader import loader
import models
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
    parser.add_argument('--model_weight', default=None,
                        help='path to model weight file to be loaded')
    parser.add_argument('--checkpoint_dir', default='checkpoints', metavar='LR',
                        help='directory path to store checkpoints')
    parser.add_argument('--model', default='maskRCNN',
                        help='which model to use for training? MaskRCNN or UNET', required=True)
    parser.add_argument('--max_instances', type=int, default=350,
                        help='maximum number of instances for maskRCNN default is 500')
    parser.add_argument('--config_path', default='config.txt',
                        help='a File containing the names of classes')

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
    # classes = utils.parse_config(args.config_path)
    # assert len(
    #     classes) == args.num_classes, "Number of classes in config and num_classes should be same"
    # d = loader.CellDataset(
    #     args.root_dir, None, args.labels_type, args.model, classes)
    # data_loader = torch.utils.data.DataLoader(
    #     d, batch_size=1, shuffle=True, num_workers=4,
    # )
    # for _ in data_loader:
    #     pass

    # model = models.get_model(
    #     args.model, args.max_instances, args.model_weight, args.num_classes)
    # if args.cuda:
    #     model.cuda()

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # music_trainer = trainer.MUSIC_ANALYSIS(
    #     model, optimizer, args.cuda, args.experiment_name, args.val_step)

    # for epoch in range(args.epochs):
    #     prev_val_acc = music_trainer.best_val_accuracy
    #     music_trainer.train(epoch, train_iterator,
    #                         val_iterator, test=False)
    #     # logger.info(
    #     #     f'Epoch {epoch}, Best Epoch {music_trainer.best_epoch},\
    #     # Best Accuracy {music_trainer.best_val_accuracy}')

    #     save_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    #     os.makedirs(save_dir, exist_ok=True)
    #     if music_trainer.best_val_accuracy > prev_val_acc:
    #         torch.save(music_trainer.best_model.state_dict(), os.path.join(
    #             save_dir, args.version+"_"+str(music_trainer.best_epoch) +
    #             "_"+str(music_trainer.best_val_accuracy)+".pt"))

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two class
    classes = utils.parse_config(args.config_path)
    # use our dataset and defined transformations
    dataset = loader.CellDataset(
        args.root_dir, utils.get_transform(train=True), args.labels_type, args.model, classes)
    dataset_test = loader.CellDataset(
        args.root_dir, utils.get_transform(train=False), args.labels_type, args.model, classes)

    indices = torch.arange(len(dataset)).tolist()
    print (len(indices))
    dataset = torch.utils.data.Subset(dataset, indices[:5])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-5:])
    print (len(dataset), len(dataset_test))
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = models.get_model(
        args.model, args.max_instances, args.model_weight, args.num_classes)
    if args.cuda:
        model.cuda()
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.05)

    # let's train it for 10 epochs
    num_epochs = 1
#     model.load_state_dict(torch.load("checkpoints/8_epoch.pt"))
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader,
                        device, epoch, print_freq=5)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        if (epoch+2) % 5 == 0:
            evaluate(model, data_loader_test, device=device)
            path = "checkpoints/"+str(epoch)+"_epoch.pt"
            torch.save(model.state_dict(), path)
    return model, data_loader_test, data_loader
    print("That's it!")


if __name__ == '__main__':
    main()
