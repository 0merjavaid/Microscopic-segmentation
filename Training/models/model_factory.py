import os
import torch
from .model import *


def get_model(name, max_instances, weights, classes):
    chosen_model = None
    if weights is not None:
        assert os.path.exists(weights)
    if name.lower() == "maskrcnn":
        assert classes >= 2
        chosen_model = get_mask_rcnn(classes, max_instances)
        if weights is not None:
            chosen_model.load_state_dict(torch.load(weights))

    elif name.lower() == "unet":
        pass

    else:
        print (name, " is currently not available, try MaskRCNN or UNET")
    return chosen_model
