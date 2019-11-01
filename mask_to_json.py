import os
import glob
import json
import cv2
import shutil
import argparse
import json2coco
import numpy as np
from utils import utils
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description='Cell instance segmentation using mask RCNN')
parser.add_argument('--root_dir',  required=True,
                    help='input batch size for training (default: 2)')
parser.add_argument('--classes', type=int, default=4, required=True,
                    help='--classes 2 if only cell and background')
parser.add_argument('--convert_to_coco', type=int, default=0,
                    help='--convert_to_coco 1, if you want to convert ')
parser.add_argument('--out_dir',  default="dataset/coco/",
                    help='--where you want to save the final coco annotations')

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
    if args.convert_to_coco:
        os.makedirs("./temp", exist_ok=True)

    for a_set in sets:
        target_json = {"shapes": []}
        set_path = os.path.join(root, a_set)
        tifs = glob.glob(os.path.join(set_path, "raw.tif"))

        pngs = glob.glob(os.path.join(set_path, "*png"))
        assert len(tifs) == 1, "Raw tif not found"
        masks = list()
        for key in classes.keys():
            class_id = classes[key]
            class_name = key + ".png" if binary != 2 else "labeled.png"
            class_label_path = os.path.join(set_path, class_name)
            assert class_label_path in pngs, class_name+" Not Found"
            mask = cv2.imread(class_label_path, -1)
            polygons = mask_to_poly(mask, binary)

            for poly in polygons:
                target_json["shapes"].append(
                    {"label": key, "points": poly})

        print ("Converting", tifs[0])
        with open(tifs[0].replace("tif", "json"), 'w') as f:
            json.dump(target_json, f)
        if args.convert_to_coco:
            im_path = "./temp/" + \
                "_".join(tifs[0].split("/")[-2:])
            target_json["imagePath"] = im_path.split("/")[-1]
            with open(im_path.replace("tif", "json"), "w") as f:
                json.dump(target_json, f)
            im = cv2.imread(tifs[0])
            cv2.imwrite(im_path.replace("json", "tif"), im)

    print ("\nConverting to COCO...")
    json2coco.process(
        **{"labels": "config.txt", "input_dir": "./temp", "output_dir": args.out_dir})
    if os.path.exists("./temp"):
        shutil.rmtree("./temp")


def mask_to_poly(mask, classes):
    points = []
    unique_colors_mask = mask.reshape(
        -1)  # if classes == 2 else mask.reshape(-1, mask.shape[2])
    unique_ids = np.unique(unique_colors_mask, axis=0)
    unique_ids = unique_ids[1:]
    # print (unique_ids)
    for uid in unique_ids:
        segment = np.zeros_like(mask)
        # uid = uid #if classes != 2 else np.array([uid] * 3)
        # print (uid, segment.shape)
        segment[mask == uid] = 1
        segment = segment.astype("uint8")
        # print (np.sum(segment))

        contours, hierarchy = cv2.findContours(
            segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 1:
            max_area = 0
            bigges_cnt = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    bigges_cnt = cnt
            contours = [bigges_cnt]

        contours = np.array(contours).reshape(-1, 2)  # .tolist()
        if contours.shape[0] <= 2:
            continue
        contours = contours.tolist()
        points.append(contours)
    # dell = np.zeros_like(mask).astype("uint8")
    # for point in points:
    #     dell = cv2.fillPoly(dell, pts=[point], color=255)
    # print (np.sum(dell))
    # cv2.imwrite("img.jpg", dell.astype("uint8"))
    # 0/0
    return points


initialize_pairs(args.root_dir, args.classes)
