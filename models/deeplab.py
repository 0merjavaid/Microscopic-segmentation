import torch
import torchvision
import torch.nn as nn


def get_deeplab(backbone="resnet101", pretrained=True):
    if backbone == "resnet50":
        model = torchvision.models.segmentation.deeplabv3_resnet50(
            pretrained=pretrained)
    elif backbone == "resnet101":
        model = torchvision.models.segmentation.deeplabv3_resnet101(
            pretrained=pretrained)

    model.classifier[4] = nn.Conv2d(
        in_channels=256,
        out_channels=2,
        kernel_size=1,
        stride=1
    )
    for i, param in enumerate(model.parameters()):
        if i < 250:
            param.requires_grad = False

    return model
