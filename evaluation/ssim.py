"""
    SSIM metric
    Created by Jet-Tsyn Lee 01/08/2018
    Last Update v0.6 31/08/2018

    Evaluates images against the SSIM metric
"""

import warnings


from skimage.measure import compare_ssim
from skimage.transform import resize
from scipy.misc import imsave
from scipy.ndimage import imread
import numpy as np
import cv2
from glob import glob
import os
import time



warnings.filterwarnings('ignore')

# specify resized image sizes
height = 2**10
width = 2**10



def get_img(path, norm_exposure=False):

    img = imread(path, flatten=True).astype(int)
    # resizing returns float vals 0:255; convert to ints for downstream tasks
    img = resize(img, (height, width), anti_aliasing=True, preserve_range=True)


    if norm_exposure:
        img = normalize_exposure(img)
    return img



def get_histogram(img):
    h, w = img.shape
    hist = [0.0] * 256
    for i in range(h):
        for j in range(w):
            hist[img[i, j]] += 1
    return np.array(hist) / (h * w)


def normalize_exposure(img):

    img = img.astype(int)
    hist = get_histogram(img)

    # get the sum of vals accumulated by each position in hist
    cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
    # determine the normalization values for each unit of the cdf
    sk = np.uint8(255 * cdf)
    # normalize each position in the output image
    height, width = img.shape
    normalized = np.zeros_like(img)

    for i in range(0, height):
        for j in range(0, width):
            normalized[i, j] = sk[img[i, j]]
    return normalized.astype(int)




# ##########  EVALUATION METHODS  ##########
# =====  STRUCTURAL SIMILARITY INDEX  =====
def structural_sim(path_a, path_b):
    img_a = get_img(path_a)
    img_b = get_img(path_b)
    sim, diff = compare_ssim(img_a, img_b, full=True)
    return sim, diff


# #########  MAIN  ##########
if __name__ == '__main__':

    log_date = time.strftime("%d%b%Y%H%M%S", time.localtime(time.time()))

    # File Path
    file_dir = "./Files"    # Main dataset folder containing the images
    log_dir = "./Results"   # Results location
    res_dir = os.path.join(log_dir,log_date)

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)


    # Folders to compare, requires two arrays for fake imafes and real images to compare, order of array elements must match for both arrays
    #fake_dir = ["22t_fake","23t_fake","35t_fake","50t_fake","57t_fake","65t_fake","75t_fake","87t_fake","91t_fake","100t_fake","107t_fake",
    #            "22v_fake","23v_fake","35v_fake","50v_fake","57v_fake","65v_fake","75v_fake","87v_fake","91v_fake","100v_fake","107v_fake"]
    #real_dir = ["22t_real","23t_real","35t_real","50t_real","57t_real","65t_real","75t_real","87t_real","91t_real","100t_real","107t_real",
    #            "22v_real","23v_real","35v_real","50v_real","57v_real","65v_real","75v_real","87v_real","91v_real","100v_real","107v_real"]

    #fake_dir = ["t_512", "t_batch", "t_cont","t_data","t_iter","t_l1","t_sel_no","t_sel_noise",
    #            "v_batch","v_cont","v_data","v_iter","v_l1","v_512"]
    #real_dir = ["11t_512_test","t_test","t_test","t_test","t_test","t_test","t_test","t_test",
    #            "v_test","v_test","v_test","v_test","v_test","11v_512_test"]

    fake_dir = ["t_512","v_512"]
    real_dir = ["11t_512_test","11v_512_test"]


    # Loop all directories
    for i in range(len(fake_dir)):

        # Log paths
        inv_path = os.path.join(res_dir, fake_dir[i]+log_date+"_individual_results.txt")
        tot_path = os.path.join(res_dir, fake_dir[i]+log_date+"_total_results.txt")

        # Path of images
        fake_dataset = glob(os.path.join(file_dir,fake_dir[i],"*"))
        real_dataset = glob(os.path.join(file_dir,real_dir[i],"*"))

        fake_total = len(fake_dataset)
        real_total = len(real_dataset)
        

        # Loop all images in Fake directory
        for fake_file in fake_dataset:
            
            total_ss = 0

            if ".jpg" not in fake_file:
                continue

            # Loop all images in real dataset
            for real_file in real_dataset:
                if ".jpg" not in real_file:
                    continue

                # Calcuate SSIM of fake image vs real image
                sim, diff = structural_sim(fake_file, real_file)

                # Save to log
                with open(inv_path, "a") as inv_txt:
                    inv_txt.write("\n" + fake_file + " vs " + real_file
                            + ", Structural Similarity: " + str(sim)
                            )

                total_ss += sim # sum score for mean

            # Calculage mea similarity for fake image
            mean_ss = total_ss / real_total

            # print to log
            print("Image %s =" % (fake_file), "Structural Similarity: ",str(mean_ss), )
            with open(tot_path, "a") as inv_txt:
                inv_txt.write("\n" + fake_file + " - "
                        + ", Mean Structural Similarity: " + str(mean_ss)
                        )



