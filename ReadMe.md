# Multi-detector

Multi detector is project that run multiple models of object detection , we've download the Swin-Transformer-Object-Detection and Detectron2.
The project allows you to run detection , filter the results by score or classes , and convert it to Coco Format.


## 1. Environments installation


### Detectron2

#### Requirements
	Linux or macOS with Python ≥ 3.7
	gcc & g++ ≥ 5.4 are required

#### Installation

```
conda create --name detectron2 python==3.8
conda activate detectron2
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -r requirements/requirements_detectron2.txt
python prosegur_detection/models/detectron2/setup.py
```		
### Swin

#### Requirements
	Linux or macOS (Windows is in experimental support)
	Python 3.6+
	PyTorch 1.3+
	CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
	GCC 5+
	MMCV

#### Installation
```
conda create --name swin python==3.8
conda activate swin
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.1/index.html
pip install -r requirements/requirements_swin.txt
pip install -v -e .  # or "python setup.py develop"
```		
 Once the environments installed, git clone the detectron2 and swin inside models
```
cd models
git clone https://github.com/facebookresearch/detectron2
git clone https://github.com/microsoft/Swin-Transformer
```

## 2. Structure

The architecture of the project
```
  prosegur
	├── 100_coco
	├── config
	├── logs
	├── weights
	├── requirements
	├── models
	|    ├── Swin
	|    └── detectron2
	├── output
	|    ├── detectron2
	|    │   ├── annotations
	|    │   ├── images
	|    │   └── videos
	|    └── swin
	|	 ├── annotations
	|	 ├── images
	|	 └── videos
	├── pipeline_detect_and_filter.py
	├── pipeline_filter.py
	├── detection.py
	├── utils
	└── ReadMe.md
```		

## 3. Get started

Don't forget Every time you want to run detections of particular model, you have to activate the right environments with conda activate

### Detectron2

For using detectron2 model choose config file from prosegur_detection/models/detectron2/config, download corresponding weights from https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md and save into prosegur/weights

1. Using detectron2 model on 100 images from person-pet dataset , the annotations result are stored in output/detectron2/annotations/ and the images will be stored in output/detectron2/images

```
python detection.py 100_coco/images/ models/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml output/ detectron2 0.2 --opts MODEL.WEIGHTS weights/model_final_f10217.pkl
```		

2. Using detectron2 model on video, showing detection on screen image per image and save it into outpdir 	

```
python detection.py 100_coco/images/ models/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml output/ detectron2 0.2 --video Peoplewalking.mp4 --output_video True --opts MODEL.WEIGHTS weights/model_final_f10217.pkl
```		

### Swin

For using swin transformer model choose config file from prosegur_detection/models/swin/configs, download corresponding weights from https://github.com/microsoft/Swin-Transformer and save into prosegur/weights

1. Using swin model on 100 images from person-pet dataset, the annotations result are stored in output/swin/annotations/ and the images will be stored in output/swin/images

```
python detection.py 100_coco/images/ models/Swin/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py --checkpoint weights/mask_rcnn_swin_tiny_patch4_window7_1x.pth output/ swin 0.2
```		

2. Using swin model with pipeline_detect_filter.py, to launch detection, filter it with score_threshold = 0.9, keep only classes [dog, cat, person] and convert it to coco annotations   

```
python pipeline_detect_and_filter.py 100_coco/images/ models/Swin/configs/swin/mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py output/
swin 0.2 0.9 --class_filter dog cat person --checkpoint weights/mask_rcnn_swin_tiny_patch4_window7_1x.pth
```

3. It's also possible to run only filtering and converting with pipeline_filter.py, by giving in args the path to detection results

```
python pipeline_filter.py 0.7 output/detectron2/annotations/annotations_csv/results.csv --class_filter person dog cat
```
