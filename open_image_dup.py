import logging
import os
import pandas as pd
import json
import time
import collections
from argparse import ArgumentParser
import logging
from datetime import datetime

"""
Para el dataset OpenImages: utilidad que devuelva las imágenes de OpenImages que se encuentran en más de una carpeta

Input: path donde encontrar todos los datasets de OpenImages y el prefijo de los datasets de interés (dentro de la ruta
 que pasemos, buscar en todas las carpetas que tengan el prefijo “open-images-v6“)
 
 use set.intersection(set(os.lisdir(cat,dog) ))with filename
Salida: CSV con 4 columnas: filename, path, categoría numérica de OpenImages y label alfanumérica (person, cat, dog)"""

date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT,
                    filename=f"logs/logs_dup/log_{date}",
                    filemode='a',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


def function_args():
    parser = ArgumentParser()
    parser.add_argument('open_images_path', help='path where all open images path are', type=str)
    parser.add_argument('open_images_prefixe', help='prefixe of open_images dataset', type=str)
    parser.add_argument('open_images_sufixe', help='sufixe of open_images dataset ', type=str)
    parser.add_argument('path_label_ID', help='path to file that contains label ID ', type=str)
    args = parser.parse_args()
    return args


def get_path(split, path_oi, prefixe, suffixe):
    """
    Get all paths of split (train, test, val) for all open-images dataset (cat,person,dog)
    Args:
        split: the split to look into (train,test,val)
        path_oi: path to the directory containing all open images dataset
        prefixe: prefixe of open images dataset
        suffixe: suffixe of open images dataset

    Returns:
        path to all open-images dataset (cat/person/dog) for one split
    """
    path_cat = os.path.join(os.path.join(path_oi, prefixe + '_cat'), suffixe)
    path_dog = os.path.join(os.path.join(path_oi, prefixe + '_dog'), suffixe)
    path_person = os.path.join(os.path.join(path_oi, prefixe + '_person'), suffixe)

    path_cat = os.path.join(os.path.join(path_cat, split), 'cat')
    path_dog = os.path.join(os.path.join(path_dog, split), 'dog')
    path_person = os.path.join(os.path.join(path_person, split), 'person')
    return path_dog, path_person, path_cat


def get_intersection(path_cat, path_dog, path_person):
    """
    get all filenames that are in more than one dataset
    Args:
        path_cat: path to open-images-cat images
        path_dog: path to open-images-dog images
        path_person: path to open-images-person images

    Returns: all intersection combination as list that contains images that are in more than one dataset
    """
    # list filenames in all dataset and convert it into a set
    set_cat = set(os.listdir(path_cat))
    set_person = set(os.listdir(path_person))
    set_dog = set(os.listdir(path_dog))
    # get intersection between the different set
    inter_dog_cat = set_dog.intersection(set_cat)
    inter_dog_person = set_dog.intersection(set_person)
    inter_cat_person = set_cat.intersection(set_person)
    inter_dog_cat_person = inter_dog_cat.intersection(set_person)
    list_intersection = [inter_dog_cat, inter_dog_person, inter_cat_person, inter_dog_cat_person]
    # remove directory "labels" in every set
    for inter in list_intersection:
        if 'labels' in inter:
            inter.remove('labels')
    # remove images that are in more than 2 dataset
    # explain : filename that are in inter_dog_cat_person are also in dog_cat , dog_person, cat_person, remove to avoid duplicate
    inter_dog_cat = inter_dog_cat - inter_dog_cat_person
    inter_dog_person = inter_dog_person - inter_dog_cat_person
    inter_cat_person = inter_cat_person - inter_dog_cat_person

    return [list(inter_dog_cat), list(inter_dog_person), list(inter_cat_person), list(inter_dog_cat_person)]


def path_column(row, path_dog, path_cat, path_person):
    # build the column path
    if row['label'] == ['cat', 'person']:
        return [os.path.join(path_cat, row['filename']), os.path.join(path_person, row['filename'])]
    elif row['label'] == ['person', 'dog']:
        return [os.path.join(path_dog, row['filename']), os.path.join(path_person, row['filename'])]
    elif row['label'] == ['cat', 'dog']:
        return [os.path.join(path_cat, row['filename']), os.path.join(path_dog, row['filename'])]
    elif row['label'] == ['cat', 'dog', 'person']:
        return [os.path.join(path_cat, row['filename']), os.path.join(path_dog, row['filename']),
                os.path.join(path_person, row['filename'])]


def label_id_column(row, label_id):
    # build the column ID_label
    if row['label'] == ['cat', 'person']:
        return [label_id[0], label_id[2]]
    elif row['label'] == ['person', 'dog']:
        return [label_id[1], label_id[2]]
    elif row['label'] == ['cat', 'dog']:
        return [label_id[0], label_id[1]]
    elif row['label'] == ['cat', 'dog', 'person']:
        return label_id


def build_dataframe(df, intersection, path_dog, path_cat, path_person, label_list, label_id):
    # build the dataframe of duplicate
    df['filename'] = pd.Series(intersection)
    df['label'] = pd.Series([label_list] * len(intersection))
    df['path'] = df.apply(lambda row: path_column(row, path_dog, path_cat, path_person), axis=1)
    df['ID_label'] = df.apply(lambda row: label_id_column(row, label_id), axis=1)
    return df


def main(split, path_oi, prefixe, suffixe, label_id):
    """
    Construct the dataframe that will have all images that are in more than one dataset
    Args:
        split: the split to look into (train,test,val)
        path_oi: path to the directory containing all open images dataset
        prefixe: prefixe of open images dataset
        suffixe: suffixe of open images dataset
        label_id: the ID of the label

    Returns:
        return a dataframe with all the duplicate or none if no duplicate were found
    """
    # get the path of all dataset of the split given as input
    path_dog, path_person, path_cat = get_path(split, path_oi, prefixe, suffixe)
    # get all combination of intersection between the differents set of images
    inter_dog_cat, inter_dog_person, inter_cat_person, inter_dog_cat_person = get_intersection(path_cat, path_dog,
                                                                                               path_person)
    list_df = []
    # build dataframe for all combination of intersection
    if len(inter_dog_cat) > 0:
        df_dog_cat = pd.DataFrame(columns=['filename', 'path', 'label', 'ID_label'])
        list_df.append(
            build_dataframe(df_dog_cat, inter_dog_cat, path_dog, path_cat, path_person, ['cat', 'dog'], label_id))
    if len(inter_dog_person) > 0:
        df_dog_person = pd.DataFrame(columns=['filename', 'path', 'label', 'ID_label'])
        list_df.append(
            build_dataframe(df_dog_person, inter_dog_person, path_dog, path_cat, path_person, ['person', 'dog'],
                            label_id))
    if len(inter_cat_person) > 0:
        df_cat_person = pd.DataFrame(columns=['filename', 'path', 'label', 'ID_label'])
        list_df.append(
            build_dataframe(df_cat_person, inter_cat_person, path_dog, path_cat, path_person, ['cat', 'person'],
                            label_id))
    if len(inter_dog_cat_person) > 0:
        df_dog_cat_person = pd.DataFrame(columns=['filename', 'path', 'label', 'ID_label'])
        list_df.append(
            build_dataframe(df_dog_cat_person, inter_dog_cat_person, path_dog, path_cat, path_person,
                            ['cat', 'dog', 'person'], label_id))
    # concat all combination into one dataframe
    if len(list_df) > 0:
        result = pd.concat(list_df, ignore_index=True)
        return result, True
    else:
        return None, False


if __name__ == '__main__':
    args = function_args()
    path_oi = args.open_images_path
    prefixe = args.open_images_prefixe
    suffixe = args.open_images_sufixe
    # load dataframe that contains mapping ID to label
    df_label_id = pd.read_csv(args.path_label_ID, names=['ID', 'label'])
    # get category ID of dog person and cat label
    cat_id = df_label_id[df_label_id['label'] == 'Cat']['ID'].values[0]
    dog_id = df_label_id[df_label_id['label'] == 'Dog']['ID'].values[0]
    person_id = df_label_id[df_label_id['label'] == 'Person']['ID'].values[0]
    cat_dog_person_id = [cat_id, dog_id, person_id]
    # for each split ( train,test,val) get a dataframe of all images having more than one category
    res_train, train_bool = main('train', path_oi, prefixe, suffixe, cat_dog_person_id)
    res_test, test_bool = main('test', path_oi, prefixe, suffixe, cat_dog_person_id)
    res_val, val_bool = main('validation', path_oi, prefixe, suffixe, cat_dog_person_id)
    list_results = []
    if val_bool:
        list_results.append(res_val)
    if train_bool:
        list_results.append(res_train)
    if test_bool:
        list_results.append(res_test)
    if len(list_results) > 0:
        result = pd.concat(list_results, ignore_index=True)
        result.to_csv('duplicate_open/open_images_dup.csv', header=['filename', 'path', 'label', 'ID_label'],
                      index=False)
    else:
        logging.info('final dataframe empty no duplicate found')
