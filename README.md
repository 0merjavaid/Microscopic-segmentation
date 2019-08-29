# Microscopic Cell Segmentation

Microscopic cell segmentation is done using Mask RCNN and UNET. Both Training and Inference scripts are implemented

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Installing

What things you need to install the software and how to install them

```
git clone https://github.com/0merjavaid/microscopic-segmentation.git
pip install -r requirements.txt
```
### Download Data

Training and Test data can be downloaded from dropbox or run the following script

```
cd dataset
./data.sh
```

### Data Conversion

There are two kinds of data formats used in this project. i) Labelme ii) image pairs. To convert Image pairs to Coco style and trainable format of repository run the following command

```
cd microscopic-segmentation
nano config.txt 
#Add names of all the classes in seperate lines except background and save
python mask_to_json.py --root_dir dataset/image_pairs --classes 4 --convert_to_coco 1 

# to convert directly from labelme format to coco run the following command
python json2coco.py --input_dir dataset/labelme/ --labels config.txt --output_dir ./ouput_coco/
```

### Training

The repo currently supports Training of UNET and MaskRCNN on Image pairs data format after conversion using mask_to_json.py. In order to train the models see the following sample command

```
#To Train UNET  set --model unet --lr 0.01 --num_classes 2 --epochs 100 (make sure that config.txt has only one class cell) 
python train.py --labels_type pairs --root_dir dataset/imagepairs/ --model unet --batch_size 2   --lr 0.01 --num_classes 2 --epochs 100

#To Train Mask RCNN set --model maskrcnn --lr 0.001 --num_classes 4 --epochs 20 
python train.py --labels_type pairs --root_dir dataset/imagepairs/ --model maskrcnn --batch_size 2 --epochs 20 --lr 0.001 --num_classes 4

#The weights of networks will be saved under checkpoints directory
```

### Inference

The Inference script takes A fixed folder structure and creates outputs using both UNET and MASK RCNN.


```
#To Infer using MaskRCNN and UNET see the following sample command
python infer.py --num_classes 4 --root_dir dataset/EF/ --maskrcnn_weight checkpoints/4_maskrcnn_epoch.pt --unet_weight checkpoints/49_UNET_epoch.pt  --batch_size 2

```

## Authors

* **Umer javaid** - *Initial work* - (https://github.com/0merjavaid)

See also the list of [contributors](https://github.com/0merjavaid/microscopic-segmentation/contributors) who participated in this project.
 

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

