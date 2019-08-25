import os
import glob
import json
import cv2
import argparse
import numpy as np
from utils import utils
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description='Cell instance segmentation using mask RCNN')
parser.add_argument('--root_dir',  required=True,
                    help='input batch size for training (default: 2)')
parser.add_argument('--classes', type=int, default=4, required=True,
                    help='--classes 2 if only cell and background')

args = parser.parse_args()


def initialize_pairs(root, binary):
    dataset = list()
    sets = os.listdir(root)
    sets = [i for i in sets if "set" in i]
    if binary == 2:
        classes = {"cell": 1}
    else:
        classes = utils.parse_config("config.txt")
        assert len(classes)+1 == binary
    assert len(sets) > 1, "No Sets found inside the data directory"

    for a_set in sets:
        target_json = {"shapes": []}
        set_path = os.path.join(root, a_set)
        tifs = glob.glob(os.path.join(set_path, "raw.tif"))
        print (tifs)
        pngs = glob.glob(os.path.join(set_path, "*png"))
        assert len(tifs) == 1, "Raw tif not found"
        masks = list()
        for key in classes.keys():
            class_id = classes[key]
            class_name = key + ".png" if binary != 2 else "instances_ids.png"
            class_label_path = os.path.join(set_path, class_name)
            assert class_label_path in pngs, class_name+" Not Found"
            mask = cv2.imread(class_label_path)
            polygons = mask_to_poly(mask, binary)

            for poly in polygons:
                target_json["shapes"].append(
                    {"label": key, "points": poly})
        with open(tifs[0].replace("tif", "json"), 'w') as f:
            json.dump(target_json, f)


def mask_to_poly(mask, classes):
    points = []
    unique_colors_mask = mask.reshape(
        -1) if classes == 2 else mask.reshape(-1, mask.shape[2])
    unique_ids = np.unique(unique_colors_mask, axis=0)
    unique_ids = unique_ids[1:]
    bin_mask = np.zeros((mask.shape[0], mask.shape[1]))
    for uid in unique_ids:
        segment = mask.copy()
        uid = uid if classes != 2 else np.array([uid] * 3)

        segment = cv2.inRange(segment, uid, uid)

        contours, hierarchy = cv2.findContours(
            segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 1:
            contours = [cnt.reshape(1, -1, 2).tolist() for cnt in contours]
        else:
            contours = np.array(contours).reshape(1, -1, 2).tolist()

        points.append(contours)
    return points


initialize_pairs(args.root_dir, args.classes)
