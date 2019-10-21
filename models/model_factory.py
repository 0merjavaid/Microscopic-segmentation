import os
import torch
import torch.nn as nn
from .maskrcnn import *
from .unet import Unet34
from .deeplab import *
import torchvision.models as models


def get_model(name, weights, classes=4, max_instances=250, maskrcnn_backbone="resetnet101"):
    chosen_model = None
    if weights is not None:
        assert os.path.exists(weights)
    if name.lower() == "maskrcnn":
        assert classes >= 2
        chosen_model = get_mask_rcnn(classes, max_instances, maskrcnn_backbone)

    elif name.lower() == "unet":
        m_base = nn.Sequential(
            *(list(models.resnet34(pretrained=True).children())[:8]))
        chosen_model = Unet34(m_base)

    elif name.lower() == "deeplab":
        chosen_model = get_deeplab()

    else:
        print (name, " is currently not available, try MaskRCNN, UNET or Deeplab")

    if weights is not None:
        assert os.path.exists(weights)
        chosen_model.load_state_dict(torch.load(weights))
    return chosen_model
