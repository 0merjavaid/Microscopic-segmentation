import cv2
import dice
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.utils import *
import torch.nn.functional as F
from mask_utils.engine import train_one_epoch, evaluate


class TrainModel:

    def __init__(self, model, optimizer, model_type, cuda=False, val_step=50):
        self.cuda = cuda
        self.model = model
        self.best_score = 0
        self.best_model = None
        self.val_step = val_step
        self.optimizer = optimizer
        self.model_type = model_type
        self.criterion = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

    def predict(self, images, labels):
        if self.cuda:
            images = torch.cat(images, 0).cuda().float()
            if self.model_type == "deeplab":
                labels = torch.cat(labels, 0).cuda().long()
            else:
                labels = torch.cat(labels, 0).cuda().float()
        else:
            images = images.float()
            labels = labels.float()
        self.optimizer.zero_grad()
        if self.model_type == "deeplab":
            output = self.model(images)["out"]
            loss = self.ce(output, labels)
        else:
            output = self.model(images)
            loss = self.criterion(output, labels)
        return output, loss, labels

    def train_unet(self, epoch, data_iterator, val_iterator):
        self.model.train()
        train_loss, train_accuracy = list(), list()
        best_model = None
        best_val_accuracy = 0
        best_epoch = 0
        if (epoch+1) % 15 == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] /= 1.5
                print ("Changing Learning Rate:,", param_group['lr'])
        for i, (images, labels)in enumerate(tqdm(data_iterator)):
            labels = [label["bin_mask"] for label in labels]
            output, loss, labels = self.predict(
                images, labels)

            train_loss.append(loss.cpu().item())
            if i % self.val_step == 0:
                avg_loss = np.mean(train_loss[-10:])
                print(f'Epoch" {epoch}, Iter: {i},TRAINING__   loss :{loss}, , smooth_loss: {avg_loss}')
                self.best_model = self.model

            loss.backward()
            self.optimizer.step()

    def train_maskrcnn(self, data_loader, test_data_loader,
                       epoch, print_freq=5):
        train_one_epoch(self.model, self.optimizer, data_loader,
                        self.cuda, epoch, print_freq=print_freq)
        if epoch % 1 == 0:
            evaluate(self.model, test_data_loader, device=self.cuda)
            path = "checkpoints/"+str(epoch)+"_maskrcnn_epoch.pt"
            torch.save(self.model.state_dict(), path)

    def train(self, epoch, train_iterator, test_iterator):
        if self.model_type.lower() == "unet" or self.model_type.lower() == "deeplab":
            self.train_unet(epoch, train_iterator, test_iterator)
            if epoch % 1 == 0:
                score = self.test(test_iterator, False)
                path = "checkpoints/"+self.model_type+".pt"
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = self.model
                    torch.save(self.model.state_dict(), path)

        else:
            self.train_maskrcnn(train_iterator, test_iterator, epoch)

    def test(self,  val_iterator, save_images=False):
        print ("\nTesting ...")
        self.model.eval()
        summ = 0
        total_images = 0
        for i, (images, labels)in enumerate(tqdm(val_iterator)):
            labels = [label["bin_mask"] for label in labels]
            total_images += 1
            with torch.no_grad():
                output, loss, labels = self.predict(
                    images, labels)
                if self.model_type == "deeplab":
                    output = torch.max(output, 1)[1]
                output = (output > 0.55).float()
                score = dice.dice_coeff(output, labels.float())
                summ += score

                if save_images or True:
                    output = output.cpu().detach().numpy().squeeze()
                    im = np.transpose(images[0].cpu().detach(
                    ).numpy().squeeze(), (1, 2, 0))
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    output[output > 0.5] = 255
                    output[output <= 0.5] = 0
                    output = np.hstack(
                        (np.hstack((im*255, output)), labels.cpu()[0].detach().numpy().squeeze()*255)).astype("uint8")

                    cv2.imwrite("results/"+str(i) +
                                ".jpg", output)
        dice_score = float(summ)/total_images
        print ("Test DICE Coefficeint = ", dice_score)
        return dice_score

    def eval_metrics(self, pred, target):
        return float(torch.sum(torch.max(pred, 1)[1] == target))/pred.shape[0]
