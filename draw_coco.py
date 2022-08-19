import os
from argparse import ArgumentParser
from ast import literal_eval
import json
from pathlib import Path
import cv2
import pandas as pd
import logging
from datetime import datetime
import sys

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT,
                    filename=f"logs/logs_coco/log_draw_{date}",
                    filemode='a',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


def draw_args():
    parser = ArgumentParser()
    parser.add_argument('annot_path', help='path to annotation at coco format', type=str)
    parser.add_argument('images_dir', help='path to images', type=str)
    parser.add_argument('--output_dir', help='path to store output', type=str)
    parser.add_argument('--input_dir', help='path to images directory ', type=str)
    parser.add_argument('--from_csv', help='path to csv that contains filenames', type=str)
    parser.add_argument('--show', help='if True display image on screen', type=bool)
    parser.add_argument('--filename_list', nargs="+")

    args = parser.parse_args()
    return args


def get_annotation(img_name, images_coco, annot_coco):
    # get all image_id for same filename
    row_img = images_coco[images_coco['file_name'] == img_name]
    if len(row_img.index) == 1:
        id_img = row_img.loc[row_img.index[0], 'id']
        # get annotations
        annot = annot_coco[annot_coco['image_id'] == id_img]
    else:
        # if len >1 means that there is multiple id for the same filename (duplicate)
        # merge annotations of all id
        list_id = set(row_img['id'])
        annot = pd.DataFrame(
            columns=['segmentation', 'iscrowd', 'area', 'image_id', 'bbox', 'category_id', 'id'])
        for i in list_id:
            annot = pd.concat([annot, annot_coco[annot_coco['image_id'] == i]])
    if len(annot) == 0:
        logging.info(f"image {img_name} has no annotations")
        value = False
    else:
        value = True
    return value, annot


def draw_bboxes(image, boxes, label):
    # function that draw bounding boxes onto the image given and add score, category
    [x1, y1, w, h] = [int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])]
    c1 = (x1, y1)
    c2 = (x1 + w, y1 + h)
    cv2.rectangle(image, c1, c2, (255, 0, 0), 3)
    # if the text is out of image bounding , replace it
    if y1 < 30:
        yput_text = y1 + 15
    else:
        yput_text = y1 - 10
    cv2.putText(image, label, (x1, yput_text),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)


def get_filenames(filename_list, from_csv, input_dir):
    # get the filenames list from the possible argument
    if filename_list:
        filenames = filename_list
    elif from_csv:
        filelist_csv = pd.read_csv(from_csv)
        filenames = list(filelist_csv['file_name'])
    elif input_dir:
        filenames = [file for file in os.listdir(input_dir)]
        if 'labels' in filenames:
            filenames.remove('labels')
    else:
        logging.error('no input data you have to give one of this three argument (input_dir, filename_list, from_csv')
        sys.exit()
    if len(filenames) == 0:
        logging.error("no image in directory")
        sys.exit()
    return filenames


def main(annot_path, output_dir, images_dir, filename_list, from_csv, input_dir, show):
    """
    main functions that get filenames list then load coco annotations file,
    loop into filename_list and get all annotations for the same filename taking in account the duplicate, and draw bboxes
    """
    if not os.path.isdir(images_dir):
        logging.error(f"{images_dir} does not exist")
        sys.exit()

    if output_dir and not os.path.exists(output_dir):
        print('creating dir')
        os.makedirs(output_dir)
    # get  filenames list
    filenames = get_filenames(filename_list, from_csv, input_dir)
    # load annotations
    with open(annot_path, 'r') as f:
        data = json.load(f)
    images_coco = pd.DataFrame.from_dict(data['images'])
    annot_coco = pd.DataFrame.from_dict(data['annotations'])
    categories = pd.DataFrame.from_dict(data['categories'])
    # check if images from annotations are in the directory
    for fn in filenames:
        filepath = os.path.join(images_dir, fn)
        if not os.path.isfile(filepath):
            logging.info(f"image: {fn} is not in directory {images_dir} ")
            continue
        # check if image fn has annotations, if yes get it
        value, annot = get_annotation(fn, images_coco, annot_coco)
        if value:
            image = cv2.imread(filepath)
            for index, raw in annot.iterrows():
                bboxes = raw['bbox']
                category_id = raw['category_id']
                label = categories[categories['id'] == category_id]['name'].values[0]
                draw_bboxes(image, bboxes, label)
            if output_dir:
                output_path = os.path.join(output_dir, fn)
                cv2.imwrite(output_path, image)
                # show image drawn
            if show:
                cv2.imshow('image', image)
                cv2.waitKey(0)
        else:
            logging.info(f"image: {fn} has no annotations")


if __name__ == "__main__":
    args = draw_args()
    annot_path = args.annot_path
    output_dir = args.output_dir
    images_dir = args.images_dir
    filename_list = args.filename_list
    from_csv = args.from_csv
    input_dir = args.input_dir
    show = args.show

    main(annot_path, output_dir, images_dir, filename_list, from_csv, input_dir, show)
