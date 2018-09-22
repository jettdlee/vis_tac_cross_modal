"""
Program used to generate text file for AlexNet classification
Created by Jet-Tsyn Lee 12/09/2018
"""

import os

# Folder Location path
file_path = os.path.join(".","train_data")

# Array contianing folders to include, order specifies the class number
fold_arr = ["22","23","35","50","57","65","75","87","91","100","107"]

fold_type = "v" # Type of image, v=visual, t=tactile
log_name = "train_v_gen_min.txt"    # text file name

label = 0   # starting label

# Add the generated images to the file, folders of the generated images must match the resl data
add_gen = True
gen_path = os.path.join(".","gen_data")


limit = 100000 # Apply a limit to the images added

# Loop all folders
for folder in fold_arr:
    fold_name = folder + fold_type

    ####  REAL FOLDER  ####
    fold_path = os.path.join(file_path, fold_name)
    count = 0
    # Loop all images
    for file_name in os.listdir(fold_path):
        print(file_name)

        with open(os.path.join(os.getcwd(),log_name), "a") as inv_txt:
            inv_txt.write("\n" + os.path.join(fold_path,file_name) + " " + str(label))

        count +=1
        if count >= limit:
            break
    

    ####  GENERATED FOLDER  ####
    if add_gen == True:
        gen_folder = os.path.join(gen_path, fold_name)
        count = 0
        # Loop al generated images
        for file_name in os.listdir(gen_folder):
            print(file_name)

            with open(os.path.join(os.getcwd(),log_name), "a") as inv_txt:
                inv_txt.write("\n" + os.path.join(gen_folder,file_name) + " " + str(label))

            count +=1
            if count >= limit:
                break

    # Next folder, next label
    label += 1