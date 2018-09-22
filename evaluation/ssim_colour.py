"""
    Colour-SSIM metric
    Created by Jet-Tsyn Lee 10/08/2018
    Last Update v0.6 31/08/2018

    Evaluates images against the Colour-SSIM metric
"""

import tensorflow as tf
import numpy as np
from glob import glob
import scipy
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.WARN)


class SSIM_measure:

    # Constructor
    def __init__(self, file_dir, src_arr, trg_arr, log_dir, gryscl=True, win_size=11, win_sigma= 1.5):

        self.srttime = time.strftime("%d%b%Y%H%M%S", time.localtime(time.time()))

        self.file_dir = file_dir    # Main folder containing images subfolders
        self.res_dir = os.path.join(log_dir, self.srttime+"_SSIM")  # Log folder

        if not os.path.exists(self.res_dir):
            os.makedirs(self.res_dir)

        # Array containing folders to evaluate
        self.src_dir = src_arr
        self.trg_dir = trg_arr

        # Determines if metric is measured as a greyscale image
        self.gryscl = gryscl

        # Window size of filter
        self.win_size = win_size
        self.win_sigma = win_sigma
        
    # Gaussian Window
    def gaussian_window(self):

        def _create_data(data):
            data = np.expand_dims(data, axis=-1)
            data = np.expand_dims(data, axis=-1)

            if self.gryscl == False:
                data = _add_padding(data)

            i_data = tf.constant(data, dtype=tf.float32)
            return i_data

        # Add Padding for the RGB Channels
        def _add_padding(data, pad=3):
            row, col, _, _ = data.shape
            new_shape = np.zeros([row, col, pad, pad]) 
            dif = np.array(new_shape.shape) - np.array(data.shape)
            new_data = np.pad(data,((0,dif[0]),(0,dif[1]),(0,dif[2]),(0,dif[3])), "reflect")
            
            return new_data

        x_data, y_data = np.mgrid[-self.win_size//2 + 1:self.win_size//2 + 1, 
                                    -self.win_size//2 + 1:self.win_size//2 + 1]

        x = _create_data(x_data)
        y = _create_data(y_data)

        g = tf.exp(-((x**2 + y**2)/(2.0*self.win_sigma**2)))

        return g / tf.reduce_sum(g)


    # SSIM Metric (Colour if specified)
    def tf_ssim(self, src_img, trg_img, cs_map=False, mean_metric=True):

        window = self.gaussian_window() # window shape [self.win_size, self.win_size]

        K1 = 0.01
        K2 = 0.03
        L = 1  # depth of image (255 in case the image has a differnt scale)
        C1 = (K1*L)**2
        C2 = (K2*L)**2

        mu1 = tf.nn.conv2d(src_img, window, strides=[1,1,1,1], padding='VALID')
        mu2 = tf.nn.conv2d(trg_img, window, strides=[1,1,1,1],padding='VALID')

        mu1_sq = mu1*mu1
        mu2_sq = mu2*mu2
        mu1_mu2 = mu1*mu2

        sigma1_sq = tf.nn.conv2d(src_img*src_img, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
        sigma2_sq = tf.nn.conv2d(trg_img*trg_img, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
        sigma12 = tf.nn.conv2d(src_img*trg_img, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2


        if cs_map:
            value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                        (sigma1_sq + sigma2_sq + C2)),
                    (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
        else:
            value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                        (sigma1_sq + sigma2_sq + C2))


        if mean_metric:
            value = tf.reduce_mean(value)

        return value

    # Multi scale SSIM (Not Used)
    def tf_ms_ssim(self, src_img, trg_img, mean_metric=True, level=5):
        weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
        mssim = []
        mcs = []

        for l in range(level):
            ssim_map, cs_map = self.tf_ssim(src_img, trg_img, cs_map=True, mean_metric=False)
            mssim.append(tf.reduce_mean(ssim_map))
            mcs.append(tf.reduce_mean(cs_map))
            
            filtered_im1 = tf.nn.avg_pool(src_img, [1,2,2,1], [1,2,2,1], padding='SAME')
            filtered_im2 = tf.nn.avg_pool(trg_img, [1,2,2,1], [1,2,2,1], padding='SAME')
            src_img = filtered_im1
            trg_img = filtered_im2

        # list to tensor of dim D+1
        mssim = tf.stack(mssim, axis=0)
        mcs = tf.stack(mcs, axis=0)

        value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                                (mssim[level-1]**weight[level-1]))

        if mean_metric:
            value = tf.reduce_mean(value)

        return value



    # Main
    def run(self):


        def _load_file(path_dir):
            image = scipy.misc.imread(str(path_dir), flatten=self.gryscl)
            image = np.expand_dims(image,0)
            if self.gryscl:
                image = np.expand_dims(image,-1)

            image_ph = tf.placeholder(tf.float32, shape=image.shape)

            return image, image_ph


        def _open_folder(fold_name):
            fold_arr = glob(os.path.join(self.file_dir,fold_name,"*.jpg"))
            tot = len(fold_arr)
            return fold_arr, tot

        # Loop all source images
        for i in range(len(self.src_dir)):

            # Get images in folder
            src_arr, src_tot = _open_folder(self.src_dir[i])
            trg_arr, trg_tot = _open_folder(self.trg_dir[i])

            # Log path
            inv_path = os.path.join(self.res_dir, self.src_dir[i] + self.srttime + "_individual_results.txt")
            tot_path = os.path.join(self.res_dir, self.src_dir[i] + self.srttime + "_total_results.txt")

            # Loop all images in folder
            for src_file in src_arr:

                with tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

                    #src_path = os.path.join(self.file_dir,src_dir[i],src_file)
                    src_img, src_img_ph = _load_file(src_file)

                    tot_ssim = 0
                    tot_msssim = 0

                    for trg_file in trg_arr:

                        #trg_path = os.path.join(self.file_dir, trg_dir[i], trg_file)
                        trg_img, trg_img_ph = _load_file(trg_file)


                        # Network
                        ssim_index = self.tf_ssim(src_img_ph, trg_img_ph)           # calculate SSIM score
                        #ms_ssim_index = self.tf_ms_ssim(src_img_ph, trg_img_ph)

                        # Start Session
                        
                        sess.run(tf.global_variables_initializer())
                        ssim_res = sess.run(ssim_index, feed_dict={src_img_ph: src_img, trg_img_ph: trg_img})
                        #ms_ssim_res = sess.run(ms_ssim_index, feed_dict={src_img_ph: src_img, trg_img_ph: trg_img})

                        # Print and log result
                        with open(inv_path, "a") as inv_txt:
                            inv_txt.write("\n" + src_file + " vs " + trg_file
                                    + ", SSIM: " + str(ssim_res)
                            )
                        #            + ", MS-SSIM: " + str(ms_ssim_res)
                        
                    # Sum value to calcuate mean
                    tot_ssim += ssim_res
                    #tot_msssim += ms_ssim_res
                
                # Calcuate mean SSIM score
                mean_ssim = tot_ssim / trg_tot
                #mean_ms_ssim = tot_msssim / trg_tot

                # Print and log result
                print("Image %s =" % (src_file), "SSIM: ", str(mean_ssim))#, "MS-SSIM:", str(mean_ms_ssim))

                with open(tot_path, "a") as inv_txt:
                    inv_txt.write("\n" + src_file + " - "
                            + ", Mean SSIM: " + str(mean_ssim))
                            #+ ", Mean MS-SSIM: " + str(mean_ms_ssim)
                            #)




if __name__ == "__main__":
    
    # Array must be manually setup as order of src and trg array must match
    # Folders containing fake images
    src_dir = ["22t_fake","23t_fake","35t_fake","50t_fake","57t_fake","65t_fake","75t_fake","87t_fake","91t_fake","100t_fake","107t_fake",
                "22v_fake","23v_fake","35v_fake","50v_fake","57v_fake","65v_fake","75v_fake","87v_fake","91v_fake","100v_fake","107v_fake",
                "t_batch", "t_cont","t_data","t_iter","t_l1","t_sel_no","t_sel_noise",
                "v_batch","v_cont","v_data","v_iter","v_l1","v_512","t_512"]

    # Folders containing Real images
    trg_dir = ["22t_real","23t_real","35t_real","50t_real","57t_real","65t_real","75t_real","87t_real","91t_real","100t_real","107t_real",
                "22v_real","23v_real","35v_real","50v_real","57v_real","65v_real","75v_real","87v_real","91v_real","100v_real","107v_real",
                "t_test","t_test","t_test","t_test","t_test","t_test","t_test",
                "v_test","v_test","v_test","v_test","v_test","11v_512_test","11t_512_test"]

    ssim = SSIM_measure(file_dir="./Files", 
                        src_arr=src_dir, 
                        trg_arr=trg_dir, 
                        log_dir="./Results", 
                        gryscl=False, 
                        win_size=11, 
                        win_sigma= 1.5)

    ssim.run()
