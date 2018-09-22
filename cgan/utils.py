"""
    Utilities module for cgan.py
    Created by Jet-Tsyn Lee 11/07/2018
    Last update v0.5 17/09/2018
"""

from __future__ import division

import os
import scipy.misc
import numpy as np

# Timer string
def timer(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def save_images(images, save_path, save_name, index=True):
    image = (images+1.)/2.

    h, w = image.shape[1], image.shape[2]
    c = image.shape[3]

    img = np.zeros((h, w, c))

    for idx, image in enumerate(images):
        if index:
            name = save_name+"_"+str(idx)+".jpg"
        else:
            name = save_name+".jpg"

        image_path = os.path.join(save_path, name)
        img[:h, :w, :] = image
        save_img = np.squeeze(image)
        scipy.misc.imsave(image_path, save_img)
        #print("Created image:", name)



def create_path(path):

    split_arr = path.split("/")
    path_str = ""
    for fold in split_arr:
        path_str = os.path.join(path_str, fold)
        if not os.path.exists(path_str):
            os.makedirs(path_str)
    return path_str




def select_data(in_dataset, out_dataset, batch_size, return_files = False, is_test=False):
    # Randomly select images from folder
    in_data = np.random.choice(in_dataset, batch_size)
    out_data = np.random.choice(out_dataset, batch_size)

    batch = []
    for img in range(len(in_data)):
        merge_img = merge_image(in_data[img], out_data[img], is_test=is_test)
        batch.append(merge_img)

    batch_img = np.array(batch).astype(np.float32)

    if return_files == True:
        return batch_img, in_data, out_data
    else:
        return batch_img


def merge_image(in_image, out_image,  is_test=False, flip=True):

    in_img = load_image(in_image, is_test, flip)
    out_img = load_image(out_image, is_test, flip)

    concat_img = np.concatenate((out_img, in_img), axis=2)
    return concat_img


def load_image(input_img, is_test, flip = True, float_type = False, no_noise=False):
    img = scipy.misc.imread(str(input_img))
    h = img.shape[0]
    w = img.shape[1]
    add_pad = 30    # add padding to image for additional randomization

    if is_test or no_noise:
        img = scipy.misc.imresize(img, [h, w])

    else:
        img = scipy.misc.imresize(img, [h+add_pad, w+add_pad])

        # RANDOMIZATION/NOISE
        h1 = int(np.ceil(np.random.uniform(1e-2, add_pad)))
        w1 = int(np.ceil(np.random.uniform(1e-2, add_pad)))

        img = img[h1:h1+h, w1:w1+w]

        if flip and np.random.random() > 0.5:
            img = np.fliplr(img)



    # Normalisation to bring values to [-1,1], [0,225], divide 127.5 to bring to 2.0
    img = img/127.5 - 1.

    if float_type:
        img = np.array(img).astype(np.float32)

    return img





