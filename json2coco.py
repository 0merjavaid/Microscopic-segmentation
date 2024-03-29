# !/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import cv2
import numpy as np
import PIL.Image

import labelme

try:
    import pycocotools.mask
except ImportError:
    print('Please install pycocotools:\n\n    pip install pycocotools\n')
    sys.exit(1)


def process(labels, input_dir, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(osp.join(output_dir, 'JPEGImages'), exist_ok=True)
    print('Creating dataset:', output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(labels).readlines()):
        class_id = i + 1  # starts with -1
        class_name = line.strip()
        # if class_id == -1:
        #     assert class_name == '__ignore__'
        #     continue
        class_name_to_id[class_name] = class_id
        data['categories'].append(dict(
            supercategory=None,
            id=class_id,
            name=class_name,
        ))

    out_ann_file = osp.join(output_dir, 'annotations.json')
    label_files = glob.glob(
        osp.join(input_dir.replace("images", "labels"), '*.json'))
    for image_id, label_file in enumerate(label_files):
        print('Generating dataset from:', label_file)
        with open(label_file) as f:
            label_data = json.load(f)

        base = osp.splitext(osp.basename(label_file))[0]
        out_img_file = osp.join(
            output_dir, 'JPEGImages', base + '.jpg'
        )

        img_file = osp.join(
            osp.dirname(label_file), label_data['imagePath']
        )
        img = np.asarray(PIL.Image.open(img_file))
        img = cv2.imread(img_file)
        # PIL.Image.fromarray(img).save(out_img_file)
        cv2.imwrite(out_img_file, img)
        data['images'].append(dict(
            license=0,
            url=None,
            file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
            height=img.shape[0],
            width=img.shape[1],
            date_captured=None,
            id=image_id,
        ))

        masks = []                                    # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_data['shapes']:
            points = shape['points']
            # print (points)
            # 0/0

            label = shape['label']
            shape_type = shape.get('shape_type', None)
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )
            # if label in masks:
            #     masks[label].append(masks[label][0])
            # else:
            #     masks[label] = [mask]

            points = np.asarray(points).flatten().tolist()
            masks.append([label, mask, points])
            # segmentations[label].append(points)

        for label, mask, points in masks:

            cls_name = label.split('-')[0]
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()
            bbox = cv2.boundingRect(
                np.array(points).reshape(-1, 2).astype(int))

            data['annotations'].append(dict(
                id=len(data['annotations']),
                image_id=image_id,
                category_id=cls_id,
                segmentation=points,
                area=area,
                bbox=bbox,
                iscrowd=0,
            ))

    with open(out_ann_file, 'w') as f:
        json.dump(data, f)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input_dir', help='input annotated directory')
    parser.add_argument('--output_dir', help='output dataset directory')
    parser.add_argument('--labels', help='labels file', required=True)
    args = parser.parse_args()

    # if osp.exists(args.output_dir):
    #     print('Output directory already exists:', args.output_dir)
    #     sys.exit(1)
    process(args.labels, args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()
