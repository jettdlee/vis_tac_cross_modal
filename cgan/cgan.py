"""
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Conditional Generative Adverserial Network
 	Created by Jet-Tsyn Lee 11/07/2018
	Last updated v0.5 17/09/2017

    cGAN network to generate new images from conviring a input domain to the output domain
    Additional Files:
        ops.py
        utils.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""


import os
import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import codecs
import decimal

from glob import glob
from utils import *     # Load functions from utils file
from ops import *

slim = tf.contrib.slim


# ARGUMENTS
parser = argparse.ArgumentParser(description='')
parser.add_argument('--mode', dest='mode', default='train', help='train, test')
args = parser.parse_args()



# ===============  CGANs  ===============
class CGAN:

    # ###############  INITIALIZE AND PARAMETERS  ###############
    def __init__(self, batch_size, srt_time, in_data_dir, out_data_dir, test_data_dir, result_dest, ckp_dir, load_type):

        # IMAGE SIZES
        self.height, self.width = 256, 256
        self.in_channel = 3         # Input domain depth
        self.out_channel = 3        # output domain depth
        self.output_size = 256      # Size of image output

        # TIME
        self.srt_time = srt_time
        self.file_date = time.strftime("%d%b%Y%H%M%S", time.localtime(self.srt_time))

        # NETWORK PARAMETERS
        self.kernal_size = [5, 5]
        self.strides = [2, 2]
        self.epsilon = 1e-5
        self.decay = 0.9
        self.stddev = 0.02
        self.z_dim = 100
        self.learning_rate = 2e-4

        self.l1_lmbda = 0   # l1Lambda coefficent


        # DIRECTORIES
        # Input domain directory
        self.in_data_dir = in_data_dir
        self.in_dataset = glob(os.path.join(self.in_data_dir, "*.jpg"))
        self.in_total = len(self.in_dataset)

        # Output domain directory
        self.out_data_dir = out_data_dir
        self.out_dataset = glob(os.path.join(self.out_data_dir, "*.jpg"))
        self.out_total = len(self.out_dataset)

        # Test data, images should be the same as input domain
        self.test_dir = test_data_dir
        self.test_dataset = glob(os.path.join(self.test_dir, "*.jpg"))
        self.test_total = len(self.test_dataset)

        # Output save location
        self.output_dir = create_path(result_dest)          # Folder to save generated images

        # Checkpoint
        self.ckp_dir = create_path(ckp_dir)                 # Main folder storing checkpoints
        self.log_dir = os.path.join(self.ckp_dir)           # Location for log file
        self.new_ckp = create_path(os.path.join(self.ckp_dir, "latest"))  # Location of checkpoint to load
        self.load_type = load_type                          # Variable types to load and save, see end of program

        # BATCH SIZE
        self.batch_size = batch_size
        self.batch_num = int((self.in_total+self.out_total) / self.batch_size)

        # BUILD NETWORK
        self.network()






    ####################################################################################################################
    #                                                   INITIALISE NETWORK
    ####################################################################################################################
    def network(self):

        # ==========  VARIABLES  =========
        with tf.variable_scope("Variables") as scope:
            # Placeholder for images
            self.x_placeholder = tf.placeholder(tf.float32, [self.batch_size, self.height, self.width,
                                                             self.in_channel + self.out_channel],
                                                name='real_in_and_out_images')

            # Boolean check for current training phase
            self.train_phase = tf.placeholder(tf.bool, name='is_train')

            # Placeholder for seperate images in the input and output domail
            self.x_in_dom = self.x_placeholder[:, :, :, self.in_channel:self.in_channel + self.out_channel]
            self.x_out_dom = self.x_placeholder[:, :, :, :self.in_channel]


        # ==========  GENERATOR AND DISCRIMINATOR =========
        self.G_out_x = self.generator(self.x_in_dom, self.batch_size, self.train_phase)     # Generator

        # Generator used to output test images
        self.G_output = self.generator(self.x_in_dom, self.batch_size, self.train_phase, reuse=True)

        # Concatenate images to be applied to the discriminator
        self.real_in_out = tf.concat([self.x_in_dom, self.x_out_dom], 3)    # Real in and out domain
        self.fake_in_out = tf.concat([self.x_in_dom, self.G_out_x], 3)      # Real in and fake out domain

        # Discriminator
        self.D_x, self.D_x_logits = self.discriminator(self.real_in_out, self.train_phase, reuse=False)     # Discrminator for real images
        self.D_g, self.D_g_logits_ = self.discriminator(self.fake_in_out, self.train_phase, reuse=True)     # Discrminator for fake images


        # LOSS FUNCTIONS AND ERRORS
        def sig_log(logits, labels):
            return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)

        self.D_x_loss = tf.reduce_mean(sig_log(self.D_x_logits, tf.ones_like(self.D_x)))        # Disc loss for real image
        self.D_g_loss = tf.reduce_mean(sig_log(self.D_g_logits_, tf.zeros_like(self.D_g)))      # Disc loss for fake image

        # Generator loss
        l1_reg = self.l1_lmbda * tf.reduce_mean(tf.abs(self.x_out_dom-self.G_out_x))    # Add l1 Regularisation
        self.g_loss = tf.reduce_mean(sig_log(self.D_g_logits_, tf.ones_like(self.D_g))) + l1_reg    # Generator loss

        # Discriminator loss
        self.d_loss = self.D_x_loss + self.D_g_loss


        # Vefine our optimizers and veriables to update weights
        t_vars = tf.trainable_variables()
        # Discriminator
        self.d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        # Generator
        self.g_vars = [var for var in t_vars if 'Generator' in var.name]
        # Weights and Biases only for checkpoint storage
        self.custom_save = [var for var in t_vars if '/b' in var.name
                            or '/B' in var.name
                            or '/w' in var.name
                            or '/W' in var.name
                            or '/Restore' in var.name
                            or 'Generator' in var.name]







    ####################################################################################################################
    #                                                   GENERATOR
    ####################################################################################################################
    def generator(self, image, batch_size, is_train, reuse=False):
        with tf.variable_scope("Generator") as scope:

            if reuse:
                scope.reuse_variables()

            s = self.output_size
            size_arr = [int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)]  # Images sizes

            gf_dim = 64
            channel_arr = [gf_dim, gf_dim*2, gf_dim*4, gf_dim*8, gf_dim*8, gf_dim*8, gf_dim*8, gf_dim*8]    # Dimensions
            enc_arr = []    # Create array to store for the u-net architecture


            # ~~~~~~~~~~  ENCODING  ~~~~~~~~~~
            with tf.variable_scope("encoder") as scope:
                for i in range(len(channel_arr)):
                    lbl = str(i+1)

                    # Initial layer
                    if i == 0:
                        enc_conv = conv2d(image, channel_arr[i], kernal=self.kernal_size, stride=self.strides,
                                          stddev=self.stddev, name='g_enc_conv_'+lbl)
                        enc_act = lrelu(enc_conv, name='g_enc_act_'+lbl)
                    else:
                        enc_conv = conv2d(enc_act, channel_arr[i], kernal=self.kernal_size, stride=self.strides,    # Convolution
                                          stddev=self.stddev, name='g_enc_conv_'+lbl)
                        enc_bn = batch_norm(enc_conv, epsilon=self.epsilon, decay=self.decay, name="g_enc_bn_"+lbl, # Batch Norm
                                            is_train=is_train, scale=False)
                        enc_act = lrelu(enc_bn, name='g_enc_act_'+lbl)                                              # Leaky RELU

                    # Store act to concatenate with decode
                    enc_arr.append(enc_act)

                relu = tf.nn.relu(enc_act, name="g_enc_relu")


            # ~~~~~~~~~  DECODING  ~~~~~~~~~~
            with tf.variable_scope("decoder") as scope:
                for i in range(-1,-(len(size_arr)+1),-1):
                    lbl = str(-i)
                    out_size = [batch_size, size_arr[i], size_arr[i], channel_arr[i]]

                    # Initial layer
                    if i == -1:
                        dec_conv, dec_wei, dec_bias = deconv2d(relu, out_size,kernal=self.kernal_size,              
                                                               stride=self.strides, stddev=self.stddev,
                                                               name="g_dec_conv_"+lbl, with_w=True)
                    else:
                        dec_conv, dec_wei, dec_bias = deconv2d(dec_act, out_size,kernal=self.kernal_size,           # Deconvolution
                                                               stride=self.strides, stddev=self.stddev,
                                                               name="g_dec_conv_"+lbl, with_w=True)

                    dec_bn = batch_norm(dec_conv, epsilon=self.epsilon, decay=self.decay, name="g_dec_bn_"+lbl,     # Batch Norm
                                        is_train=is_train, scale=False)
                    dec_act = tf.nn.dropout(dec_bn, 0.5)        # Activation

                    # Skip Connector
                    dec_act = tf.concat([dec_act, enc_arr[i-1]], 3)

                relu = tf.nn.relu(dec_act, name="g_dec_relu")

            # ~~~~~~~~~  OUTPUT LAYER  ~~~~~~~~~
            final_size = [batch_size, s, s, self.out_channel]
            dec_conv, dec_wei, dec_bias = deconv2d(relu, final_size, kernal=self.kernal_size, stride=self.strides,
                                                   stddev=self.stddev, name='g_dec_conv_'+str(len(size_arr)+1),
                                                   with_w=True)
            dec_act = tf.nn.tanh(dec_conv)

            return dec_act




    ####################################################################################################################
    #                                                   DISCRIMINATOR
    ####################################################################################################################
    def discriminator(self, image, is_train, reuse=False):
        with tf.variable_scope("Discriminator") as scope:
            df_dim = 64
            channel_arr = [df_dim, df_dim*2, df_dim*4, df_dim*8]    # Dimensions

            if reuse:
                scope.reuse_variables()

            # Loop each layer
            active_lay = image
            for i in range(len(channel_arr)):
                lbl = str(i+1)  # Label
                conv_lay = conv2d(active_lay, channel_arr[i], kernal=self.kernal_size, stride=self.strides,     # Convolution
                                  stddev=self.stddev,name='d_conv_'+lbl)
                bn_lay = batch_norm(conv_lay, epsilon=self.epsilon, decay=self.decay, name="d_bn_"+lbl,         # Batch Norm
                                    is_train=is_train, scale=False)

                # use conv layer for first loop
                if i == 0:
                    active_lay = lrelu(conv_lay, name='d_act_'+lbl)         # Leaky RELU
                else:
                    active_lay = lrelu(bn_lay, name='d_act_'+lbl)

            dim = int(np.prod(active_lay.get_shape()[1:]))
            fc_lay = tf.reshape(active_lay, shape=[-1, dim], name='fc1')
            weight = tf.get_variable('Weight', shape=[fc_lay.shape[-1], 1], dtype=tf.float32,
                                     initializer=norm_init(stddev=self.stddev))
            bias = tf.get_variable('Bias', shape=[1], dtype=tf.float32,initializer=cons_init(0.0))

            logits = tf.add(tf.matmul(fc_lay, weight), bias, name='logits')
            sig = tf.nn.sigmoid(logits)

            return sig, logits




    ####################################################################################################################
    #                                                   OUTPUT
    ####################################################################################################################
    # Output Test Batch results
    def training_output(self, g_loss, d_loss, iter_no=1):
        # Select Data
        gen_images, in_arr, out_arr = select_data(self.in_dataset, self.out_dataset, self.batch_size, return_files=True,
                                                  is_test=True)
        # Generate result image
        res_img = self.sess.run(self.G_output, feed_dict={self.x_placeholder: gen_images, self.train_phase: False})

        # Save path
        save_path = create_path(os.path.join(self.output_dir, self.file_date,"Iteration "+str(iter_no),"Batch"))
        img_name = self.file_date + "_iter" + str(iter_no)
        add_str = "\nInput domain: " + str(in_arr) + \
                  "\nOutput domain: " + str(out_arr) + \
                  'train:[%d], d_loss:%f, g_loss:%f' % (iter_no, d_loss, g_loss)

        # Create Log
        self.create_log(os.path.join(save_path, "log.txt"),iter_no,add_str)
        self.create_log(os.path.join(self.output_dir, self.file_date,"log.txt"), iter_no, add_str)

        # Save Image
        save_images(res_img, save_path, img_name)




    ####################################################################################################################
    #                                        GENERATE IMAGES FROM TEST DATASET
    ####################################################################################################################
    # Output Test Results
    def test_output(self,  g_loss=0, d_loss=0, iter_no=1):

        test_data = tf.placeholder(tf.float32, [1, self.height, self.width, self.in_channel], name='Test_image')
        test_generator = self.generator(test_data, 1, self.train_phase, reuse=True)

        save_path = create_path(os.path.join(self.output_dir, self.file_date, "Iteration "+str(iter_no), "Test"))

        add_str = "\nTest Data: " + str(self.test_dataset)
        if g_loss != 0 and d_loss != 0:
            add_str = add_str + "\ntrain:[%d], d_loss:%f, g_loss:%f" % (iter_no, d_loss, g_loss)
        self.create_log(os.path.join(save_path, "log.txt"), iter_no, add_str)

        for file in self.test_dataset:

            test_img = load_image(str(file), is_test=True, float_type=True)
            test_img = np.expand_dims(test_img, axis=0) # expand dims to feed into generator
            test_out = self.sess.run(test_generator, feed_dict={test_data:test_img, self.train_phase:False})
            test_name = file.split('/')[-1].split('.jpg')[0]
            save_images(test_out, save_path, test_name, index=False)



    ####################################################################################################################
    #                                           CHECKPOINTS AND OTHER
    ####################################################################################################################

    # LOAD CHECKPOINT
    def ckp_load(self):
        ckpt = tf.train.get_checkpoint_state(self.new_ckp)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.new_ckp, ckpt_name))
            return True
        else:
            return False


    # SAVE CHECKPOINT
    def ckp_save(self, iter_no, gLoss, dLoss, hist_save=False):
        # checkpoint wil save in two locations,
        # in the current date to store and act as historic
        # latest folder for code to load in future runs

        # save historic
        if hist_save:
            run_fold = create_path(os.path.join(self.ckp_dir, self.file_date)) # Date folder
            # Due to space limitations, will delete all previous checkpoints in the current run
            for file in os.listdir(run_fold):
                os.remove(run_fold+"/"+file)
            os.path.join(run_fold, "model")
            self.saver.save(self.sess, run_fold,global_step=iter_no)

        # Newest checkpoint
        # Delete all files
        for file in os.listdir(self.new_ckp):
            os.remove(self.new_ckp+"/"+file)

        save_loc = os.path.join(self.new_ckp,"model")
        print(save_loc)
        self.saver.save(self.sess, save_loc, global_step=iter_no)

        # Create log
        add_str = "\nG Loss: %d, D Loss: %d" % (gLoss, dLoss)
        self.create_log(os.path.join(self.log_dir, str(self.file_date)+"_log.txt"), iter_no, add_str)

    # Create log for current iteration
    def create_log(self, dir, iter_no, add_str = ""):
        #file = open(dir, "a")
        text_str = ["========================================",
                    "Current Run date: " + self.file_date,
                    "Iterations complete: %d" % iter_no,
                    "Current runtime: " + timer(self.srt_time, time.time()),
                    "",
                    add_str,
                    "========================================",
                    ""]

        with codecs.open(dir, "a", "utf-8") as f:
            f.write("\n".join(text_str))



    ####################################################################################################################
    #                                               TRAINING
    ####################################################################################################################
    def train(self, iterations, save_iter, ckp_iter, load_prev):


        # Create parameters log
        para_path = create_path(os.path.join(self.output_dir, self.file_date))
        with open(os.path.join(para_path, "parameters.txt"),"w") as txt:
            txt.write(
                "\n===========  TRAINING  ============"
                +"\nDate - "+ str(self.file_date)
                +"\nStart Time - " + str(time.strftime("%H:%M:%S", time.localtime(self.srt_time)))
                +"\nInput Dimension - " + str(self.in_data_dir) + ", Total Files - " + str(self.in_total)
                +"\nOutput Dimemsion - " + str(self.out_data_dir) + ", Total Files - " + str(self.out_total)
                +"\nBatch Size - " + str(self.batch_size)
                +"\nBatch Number - " + str(self.batch_num)
                +"\nIterations - " + str(iterations)
                +"\nL1 Regularization Lambda - " + str(self.l1_lmbda)
            )


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  TRAINING  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # optimiser
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            # Update network, use RMS gradient decent
            trainer_d = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.d_loss,
                                                                                             var_list=self.d_vars)
            trainer_g = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.g_loss,
                                                                                             var_list=self.g_vars)


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # START SESSION & LOAD CHECKPOINT
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        if self.load_type == "G":
            self.saver = tf.train.Saver(self.g_vars)
        elif self.load_type == "custom":
            self.saver = tf.train.Saver(self.custom_save)
        else:
            self.saver = tf.train.Saver()


        # Load Checkpoint
        if load_prev:
            chk = self.ckp_load()
            if chk:
                print("Checkpoint loaded")
            else:
                print("Checkpoint failed to load, continuing training, existing checkpoints will be deleted")


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)


        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # BEGIN LOOP
        print("\n\n################################################################################################")
        print("CGANS TRAINING")
        print("Input domain: %s, Total = %d" % (self.in_data_dir, self.in_total))
        print("Output domain: %s, Total = %d" % (self.out_data_dir, self.out_total))
        print("Iteration number: %d" % iterations)
        print("Batch size: %d, batch num per iter: %d" % (self.batch_size, self.batch_num))
        print("################################################################################################\n\n")


        for i in range(1, iterations+1):
            print("Running epoch {}/{}...".format(i, iterations))

            # Training
            for j in range(self.batch_num):
                
                # Select a batch of images
                batch_images = select_data(self.in_dataset, self.out_dataset, self.batch_size)

                # ~~~~~~~~~~  TRAIN DISCRIMINATOR  ~~~~~~~~~
                _, dLoss = self.sess.run([trainer_d, self.d_loss], feed_dict={self.x_placeholder: batch_images,
                                                                              self.train_phase: True})

                # ~~~~~~~~~~  TRAIN GENERATOR  ~~~~~~~~~
                _, gLoss = self.sess.run([trainer_g, self.g_loss], feed_dict={self.x_placeholder: batch_images,
                                                                              self.train_phase: True})

            # Print Loss
            if i % 10 == 0:
                print('train:[%d],d_loss:%f,g_loss:%f' % (i, dLoss, gLoss))

            # SAVE IMAGES
            if save_iter != 0 and i%save_iter == 0:

                # Create output from batch training
                print("Saving Batch Output")
                self.training_output(g_loss=gLoss, d_loss=dLoss, iter_no=i)

                # create output from test data
                print("Saving Test Output")
                self.test_output(g_loss=gLoss, d_loss=dLoss, iter_no=i)

            # SAVE CHECKPOINT
            if ckp_iter != 0 and (i%ckp_iter == 0 or i == iterations):
                print("Saving checkpoint")
                self.ckp_save(i, gLoss, dLoss)

        coord.request_stop()
        coord.join(threads)


    ####################################################################################################################
    #                                                    TESTING (WIP)
    ####################################################################################################################
    def test(self):

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # START SESSION & LOAD CHECKPOINT
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        if self.load_type == "G":
            self.saver = tf.train.Saver(self.g_vars)
        elif self.load_type == "custom":
            self.saver = tf.train.Saver(self.custom_save)
        else:
            self.saver = tf.train.Saver()

        # Load Checkpoint
        chk = self.ckp_load()

        if chk:
            print("Checkpoint loaded")
        else:
            print("Checkpoint failed to load, Ending TEST mode.")
            sys.exit()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        print('\n\n##################################################################################################')
        print("CGANS TESTING")
        print("Test Input: %s" % self.test_dir)
        print('Total training sample num: %d' % self.test_total)
        print('##################################################################################################\n\n')

        self.test_output()

        coord.request_stop()
        coord.join(threads)





####################################################################################################################
#                                                    MAIN
####################################################################################################################
if __name__ == "__main__":

    '''
    PARAMETERS
    batch_size - Batch size of the image array to process each iteration
    in_data_dir - dataset containing the input domain
    out_data_dir - dataset of the output domain
    test_data_dir,  directory of the test dataset, 
    result_dest, ckp_dir - directory of generated outputs, and checkpoints
    load_type - Select the checkpoint variables to load, "" = All, "G" - Generator only, "W/B" - Weights and Biases only
    '''


    srt_time = time.time() # Record run time

    # CGAN NETWORK, change directories to match location
    cgan = CGAN(batch_size=10,
                srt_time=srt_time,
                in_data_dir="./dataset/11t",
                out_data_dir="./dataset/11v",
                test_data_dir="./dataset/11t",
                result_dest="./results",
                ckp_dir="./checkpoint",
                load_type="custom"
                )


    # TRAINING
    '''
    Iterations - no of training iterations
    Save_iter - no of iterations before saving image, 0=disable
    Ckp_iter - no of iteration before saving checkpoint, 0=disable
    load_prev - load previous checkpoint
    '''
    if args.mode == "train":
        cgan.train(iterations=3000,
                   save_iter=100,
                   ckp_iter=0,
                   load_prev=False
                   )
        print("\n\n---COMPLETED TRAINING---\nRuntime:", timer(srt_time, time.time()))

    # TESTING
    elif args.mode == "test":
        cgan.test()
        print("\n\n---COMPLETE TEST---\nRuntime:", timer(srt_time, time.time()))

    # Invalid mode input
    else:
        print("Invalid mode input, please input correct type (trian, test)")


