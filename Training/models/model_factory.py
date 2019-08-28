import os
import torch
import torch.nn as nn
from .maskrcnn import *
from .unet import Unet34
import torchvision.models as models


def get_model(name, max_instances, weights, classes):
    chosen_model = None
    if weights is not None:
        assert os.path.exists(weights)
    if name.lower() == "maskrcnn":
        assert classes >= 2
        chosen_model = get_mask_rcnn(classes, max_instances)

    elif name.lower() == "unet":
        m_base = nn.Sequential(
            *(list(models.resnet34(pretrained=True).children())[:8]))
        chosen_model = Unet34(m_base)

    else:
        print (name, " is currently not available, try MaskRCNN or UNET")

    if weights is not None:
        assert os.path.exists(weights)
        chosen_model.load_state_dict(torch.load(weights))
    return chosen_model
