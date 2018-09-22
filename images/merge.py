"""
    Program to merge images to be shown
    Created by Jet-Tsyn Lee
"""



from glob import glob
import os
import numpy as np
from PIL import Image



visual_dir = "./dataset/visual"
tactile_dir = "./dataset/tactile"
output_path = "./dataset/merged"

vis_data = glob(os.path.join(visual_dir,"*.JPG"))
tac_data = glob(os.path.join(tactile_dir,"*.jpg"))
file_count = len(glob(os.path.join(output_path,"*.jpg")))
size = [1,2]

iteration = 100
for iter in range(1,iteration+1):

    vis_rand = np.random.choice(vis_data)
    tac_rand = np.random.choice(tac_data)

    list_im = [vis_rand, tac_rand]


    imgs = [Image.open(i) for i in list_im ]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

    # save that beautiful picture
    imgs_comb = Image.fromarray(imgs_comb)
    save_name = str(file_count)+".jpg"

    imgs_comb.save(os.path.join(output_path,save_name))
    print("Image created:",save_name)