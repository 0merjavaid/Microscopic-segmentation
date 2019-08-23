import dataloader
import models
# import trainer
# import utils
import argparse
import torch
from tqdm import tqdm
import os


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
    parser.add_argument('--root_directory', default='datasets/images',
                        help='directory that contains images and labels folders', required=True)
    parser.add_argument('--model_weight', default=None,
                        help='path to model weight file to be loaded')
    parser.add_argument('--checkpoint_dir', default='checkpoints', metavar='LR',
                        help='directory path to store checkpoints')
    parser.add_argument('--model', default='maskRCNN',
                        help='which model to use for training? MaskRCNN or UNET', required=True)
    parser.add_argument('--max_instances', type=int, default=350,
                        help='maximum number of instances for maskRCNN default is 500')

    args = parser.parse_args()

    if torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False

    return args


def main():
    args = get_argparser()

    model = models.get_model(
        args.model, args.max_instances, args.model_weight, args.num_classes)
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    music_trainer = trainer.MUSIC_ANALYSIS(
        model, optimizer, args.cuda, args.experiment_name, args.val_step)

    for epoch in range(args.epochs):
        prev_val_acc = music_trainer.best_val_accuracy
        music_trainer.train(epoch, train_iterator,
                            val_iterator, test=False)
        # logger.info(
        #     f'Epoch {epoch}, Best Epoch {music_trainer.best_epoch},\
        # Best Accuracy {music_trainer.best_val_accuracy}')

        save_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        if music_trainer.best_val_accuracy > prev_val_acc:
            torch.save(music_trainer.best_model.state_dict(), os.path.join(
                save_dir, args.version+"_"+str(music_trainer.best_epoch) +
                "_"+str(music_trainer.best_val_accuracy)+".pt"))


if __name__ == '__main__':
    main()
