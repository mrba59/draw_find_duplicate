# Dataset analysis and draw functions

This project aims two goals:

1. Draw bboxes on images from azure machine  with any annotations format from (yolo, open_images, coco) and taking in account the duplicate that we can found.

2. Analyze and find duplicate on open-images dataset


## Requirements
	Python ≥ 3.7
	pandas
	opencv-python


## Get started

1. Draw functions

You can pass images to be drawn, as a list, as a csv file, or as a path to images, respectively to the argument (--filename_list --from_csv --input_dir)
you can choose if you want to save or show on screen the images drawn with
--output_dir and --show

For example give filename as csv file with coco annotations:

```
python draw_coco.py path_annotations_coco path_images_dir --from_csv filename.csv
--show True --output_dir resdraw_coco/

```
with open_images annotations and filenames given as a list

```
pyhton draw_openimages.py datastore open-images-v6
--output_dir
resdraw_open --from_open_images oidv6-train-annotations-bbox.csv
--filename_list bd4e54f41185b488.jpg 0cac3c05d359ef3a.jpg 67a4071d76d28bc5.jpg
```
with yolo annotations and path to directory as input

```
pyhton draw_openimages.py datastore open-images-v6
--output_dir resdraw_open --input_dir datastore/open-images-v6-cat/raw/1/train/cat
```

2. find duplicate

This script find images that are stored in more than one open_images dataset,
(open-images-v6-cat, open-images-v6-dog, open-images-v6-person)
by given path were open images dataset are stored , the prefixe the suffixe and the mapping ID to label. it will return a csv file with all duplicated images inside duplicate_open

```
python open_images_dup.py datastotre open-images-v6 raw/1 class-descriptions-boxable.csv
```
