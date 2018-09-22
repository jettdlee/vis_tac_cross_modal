"""
    Program used to crop sections from a image
    Created by Jet-Tsyn Lee
"""


import os
import cv2

source = os.path.join(os.getcwd(),"source")  # Source Folder
directory =  os.path.join(os.getcwd(),"dataset")  # Destination folder

# Dimensions of output image, nxn
H_RESIZE = 227  
W_RESIZE = 227

# Loop file in directory
for file in os.listdir(source):

    if file.endswith(".JPG") or file.endswith(".jpg"):  # Check file is jpg

        # Load File
        img = cv2.cv2.imread(os.path.join(source, file))
        max_h = img.shape[0]
        max_w = img.shape[1]

        # Initialise Varaibles
        file_count = 0
        i_width = 0
        i_height = 0
        h_count = 0
        w_count = 0
        check_file = True

        # Loop until all sections are cropped
        while check_file:

            # get sizes of croped image from the position
            l_h = i_height + (h_count * H_RESIZE)
            u_h = l_h + H_RESIZE
            l_w = i_width + (w_count * W_RESIZE)
            u_w = l_w + W_RESIZE

            # Crop image
            crop_img = img[l_h:u_h, l_w:u_w]

            # Save cropped image, restrict saving to only the specified size size
            if crop_img.shape[1] >= W_RESIZE and crop_img.shape[0] >= H_RESIZE:
                crop_name = file[:-4] + "_" + str(file_count) + ".jpg"
                
                print("creating", crop_name, crop_img.shape[1], "x", crop_img.shape[0])

                cv2.cv2.imwrite(os.path.join(directory, crop_name), crop_img)
                file_count += 1

            w_count += 1


            # if reaches width end, reset counter and start at the beginint of next line
            if i_width + (w_count * W_RESIZE) > max_w:
                w_count = 0
                h_count += 1


            # if reaches height end, finish image
            if i_height + (h_count * H_RESIZE) > max_h:
                check_file = False