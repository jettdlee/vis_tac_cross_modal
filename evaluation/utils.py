# Utilities module for all evaluation metrics

import scipy
import numpy as np 
from glob import glob
import os
    
    
def load_image(input_img):
    img = scipy.misc.imread(str(input_img))
    h = img.shape[0]
    w = img.shape[1]
    img = scipy.misc.imresize(img, [h, w])
    img = np.array(img).astype(np.float32)
    return np.asarray(img)


def get_dataset(dir_path):
    return glob(os.path.join(dir_path, "*.jpg"))

def create_dataset(dataset):
    arr = []
    for img_file in dataset:
        img = load_image(img_file)
        arr.append(img)
    
    return arr

def create_path(path):

    split_arr = path.split("/")
    path_str = ""
    for fold in split_arr:
        path_str = os.path.join(path_str, fold)
        if not os.path.exists(path_str):
            os.makedirs(path_str)
    return path_str