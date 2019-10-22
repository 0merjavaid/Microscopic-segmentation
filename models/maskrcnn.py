import torch
import torchvision
import torch.nn as nn
import models.models_lpf.resnet as resnet
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN


def get_mask_rcnn(num_classes, max_instances, backbone="resnet101"):
    # load an instance segmentation model pre-trained pre-trained on COCO
    if backbone == "resnet50":
        print ("**************Adding Resnet 50 backbone***************")
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained=True, box_detections_per_img=max_instances)
    else:

        bb = resnet_fpn_backbone(backbone, False)
        model = MaskRCNN(bb,
                         num_classes=91, box_detections_per_img=max_instances)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model


def resnet_fpn_backbone(backbone_name, pretrained):
    if backbone_name == "resnet101_lpf":
        print ("**************Adding Resnet 101 AntiAliaing backbone***************")
        backbone = resnet.resnet101(filter_size=5)
        backbone.load_state_dict(torch.load(
            './checkpoints/resnet101_lpf5.pth.tar')['state_dict'])
    else:
        print ("**************Adding Resnet 101 backbone***************")
        backbone = torchvision.models.resnet101(pretrained=True)
    for name, parameter in backbone.named_parameters():
        if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    return_layers = {'layer1': 0, 'layer2': 1, 'layer3': 2, 'layer4': 3}

    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2,
        in_channels_stage2 * 2,
        in_channels_stage2 * 4,
        in_channels_stage2 * 8,
    ]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
