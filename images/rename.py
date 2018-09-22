"""
    Program to rename files in a folder
    Created by Jet-Tsyn Lee
"""

import os
import random


arr = ["23t","23v","35t","35v","50t","50v","57t","57v","65t","65v","75t","75v","87t","87v","91t","91v","100t","100v","107t","107v"]

for fold in arr:
    path = os.path.join(os.getcwd(),"Images",fold)

    for filename in os.listdir(path):

        extension = os.path.splitext(filename)[-1]
        rand_no = random.randint(1,10000)


        new_file_name_with_ext = str(rand_no)+extension

        while os.path.isfile(os.path.join(path,new_file_name_with_ext)) == True:
            rand_no = random.randint(1,10000)
            new_file_name_with_ext = str(rand_no)+extension

        print(os.path.join(path,filename))
        print(os.path.join(path,new_file_name_with_ext))
        os.rename(os.path.join(path,filename),os.path.join(path,new_file_name_with_ext))