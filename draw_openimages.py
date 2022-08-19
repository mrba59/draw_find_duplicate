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

"""
Para el dataset OpenImages: utilidad que pinte todos los bounding box de un filename que se pase por argumento.
Esta utilidad debe mirar en todas las carpetas con prefijo “open-images-v6“

Elegir por parámetros si queremos mostrar el resultado o no por pantalla, 
y si queremos guardar las imágenes con los bboxes dibujados en disco o no.

Entrada: un filename, una lista de filenames, un path a una carpeta con las imágenes o un fichero CSV con los filenames, 
y el prefijo del dataset donde mirar (en este caso, “open-images-v6“)
Salida: la imagen por pantalla (si se pone a True el parámetro correspondiente) y la imagen guardada en disco (si se pone a True el parámetro correspondiente)
"""

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT,
                    filename=f"logs/logs_open_images/log_draw_{date}",
                    filemode='a',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


def draw_args():
    parser = ArgumentParser()
    parser.add_argument('open_images_path', help='path where all open images path are', type=str)
    parser.add_argument('open_images_prefixe', help='prefixe of open_images dataset', type=str)
    parser.add_argument('--output_dir', help='path to store output', type=str)
    parser.add_argument('--from_open_images', help='read annotations from open_images csv file ', type=str)
    parser.add_argument('--input_dir', help='path to images directory ', type=str)
    parser.add_argument('--from_csv', help='path to csv that contains filenames', type=str)
    parser.add_argument('--show', help='if True display image on screen', type=bool)
    parser.add_argument('--filename_list', nargs="+")

    args = parser.parse_args()
    return args


def draw_bboxes(boxes, image, label):
    " function that draw bounding boxes onto the image given and add score, category"
    [x1, y1, x2, y2] = [int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])]
    c1 = (x1, y1)
    c2 = (x2, y2)
    cv2.rectangle(image, c1, c2, (255, 0, 0), 3)

    # if the text is out of image bounding , replace it
    if y1 < 30:
        yput_text = y1 + 15
    else:
        yput_text = y1 - 10
    cv2.putText(image, label, (x1, yput_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)


def construct_path(open_imesges_path, prefixe, category, split):
    # construct the path of open images with the parameter category (cat, dog,person) and split (train, test, val)
    prefixe = prefixe + '_' + category
    suffixe = os.path.join('raw/1/' + split, category)
    return os.path.join(os.path.join(open_imesges_path, prefixe), suffixe)


def check_image_in_dir(open_images_path, fn, category_list, split_list, prefixe):
    # check if the image given is present in the different dataset of open-images (cat, dog, person)/(train, test, val)
    # return a list of the different dataset taht contains the image
    list_path = []
    for category in category_list:
        for split in split_list:
            path = construct_path(open_images_path, prefixe, category, split)
            filename_path = os.path.join(path, fn)
            if os.path.isfile(filename_path):
                list_path.append(filename_path)
    return list_path


def check_yolo_annotation(list_path, fn):
    # concat all annotations from the differents dataset that contains this filename
    list_annot = []
    for path_images in list_path:
        # get labels path
        dir_label = os.path.join(os.path.dirname(path_images), 'labels')
        filename = Path(path_images).stem
        label_path = os.path.join(dir_label, filename + '.txt')
        if os.path.isfile(label_path):
            # get the annotations from label path
            with open(label_path, 'r') as f:
                annot = [line.split(' ') for line in f.read().splitlines()]
            f.close()
            for line in annot:
                list_annot.append(line)
        else:
            logging.info(f"file {label_path} does not exists")
            continue
    return list_annot


def get_filenames(filename_list, from_csv, input_dir):
    # get the filenames list from the possible argument
    if filename_list:
        filenames = filename_list
    elif from_csv:
        filelist_csv = pd.read_csv(from_csv)
        filenames = list(filelist_csv['file_name'])
    elif input_dir:
        filenames = [file for file in os.listdir(input_dir)]
        # remove directory labels
        if 'labels' in filenames:
            filenames.remove('labels')
    else:
        logging.error('no input data you have to give one of this three argument (input_dir, filename_list, from_csv')
        sys.exit()
    if len(filenames) == 0:
        logging.error("no image in directory")
        sys.exit()
    return filenames

def draw_from_yolo(fn, all_filepaths, output_dir, show, image):
    # read annotations from yolo format
    # get all annotations for this image , looking into the different dataset inside all_filepaths
    list_annot = check_yolo_annotation(all_filepaths, fn)
    if len(list_annot) > 0:
        for annot in list_annot:
            bboxe = [float(bb) for bb in annot[1:]]
            label = annot[0]
            draw_bboxes(bboxe, image, label)
        # save image drawn
        if output_dir:
            output_path = os.path.join(output_dir, fn)
            cv2.imwrite(output_path, image)
            # show image drawn
        if show:
            cv2.imshow('image', image)
            cv2.waitKey(0)
    else:
        logging.info(f"image {fn} has no annotations in any dataset of {prefixe}")

def check_open_annotation(annotations,fn):
    # check in train/test/val annotations file, if filename has annotation
    value = True
    if fn not in list(annotations['ImageID']):
        logging.error(f"image: {fn} has no annotations")
        value = False

    return value

def draw_from_open_images(annotation, fn, mapping_label, output_dir, show, image):
    # check annotations for filename
    fn = os.path.splitext(fn)[0]
    value = check_open_annotation(annotation,fn)

    if value:
        h, w, _ = image.shape
        # get all annotations for one filename
        annot = annotation[annotation['ImageID']==fn]
        for index, row in annot.iterrows():
            # get bboxes label and draw
            bboxe = [float(row['XMin'])*w, float(row['YMin'])*h, float(row['XMax'])*w, float(row['YMax'])*h]
            label = mapping_label[mapping_label['ID']== row['LabelName']]['label'].values[0]
            draw_bboxes(bboxe, image, label)
        # save image drawn
        if output_dir:
            output_path = os.path.join(output_dir, fn+'.jpg')
            cv2.imwrite(output_path, image)
            # show image drawn
        if show:
            cv2.imshow('image', image)
            cv2.waitKey(0)
    else:
        logging.error(f"image: {fn} has no annotations")

def main(open_images_path, output_dir, filename_list, from_csv, input_dir, show, prefixe, from_open_images):
    # main functions that get filenames list,  then loop into it
    #  for each image get all paths of open-images dataset tant contain it
    # finally choose whereas read from yolo or open images annotations and draw it
    if output_dir and not os.path.exists(output_dir):
        print('creating dir')
        os.makedirs(output_dir)
    if from_open_images:
        # load the annotations file from open images and the mapping ID to label
        open_images_annot = pd.read_csv(from_open_images, header=0)
        mapping_label = pd.read_csv('class-descriptions-boxable.csv', names=['ID', 'label'])
    # get image file names
    filenames = get_filenames(filename_list, from_csv, input_dir)
    for fn in filenames:
        # check if filename is in more than one dataset of open images (cat, dog, person)
        all_filepaths = check_image_in_dir(open_images_path, fn, ['cat', 'dog', 'person'],
                                           ['train', 'test', 'validation'], prefixe)

        if len(all_filepaths) == 0:
            logging.error(f"image: {fn} is not in any directory of {prefixe}")
            continue
        logging.info(f"image {fn} is in those dataset {all_filepaths}")
        # get filepath to load image
        filepath = all_filepaths[0]
        image = cv2.imread(filepath)
        # read annotations from yolo format
        if from_open_images:
            draw_from_open_images(open_images_annot, fn, mapping_label, output_dir, show, image)
        else:
            draw_from_yolo(fn, all_filepaths, output_dir, show, image)

if __name__ == "__main__":

    args = draw_args()
    open_images_path = args.open_images_path
    output_dir = args.output_dir
    show = args.show
    filename_list = args.filename_list
    from_csv = args.from_csv
    input_dir = args.input_dir
    prefixe = args.open_images_prefixe
    from_open_images = args.from_open_images

    main(open_images_path, output_dir, filename_list, from_csv, input_dir, show, prefixe,from_open_images)
