"""
    Operations module for cgan.py
    Created by Jet-Tsyn Lee 11/07/2018
    Last update v0.5 17/09/2018
"""

import tensorflow as tf
import numpy as np



# activation
def lrelu(x, name="lrelu", leak=0.2):
    return tf.maximum(x, leak * x, name=name)


#convolution
def conv2d(input_, output_dim, kernal=[5,5], stride=[2,2], stddev=0.02, name=None):
    with tf.variable_scope(name):
        w = tf.get_variable('Weight', [kernal[0], kernal[1], input_.get_shape()[-1], output_dim], initializer=norm_init(stddev=stddev))
        biases = tf.get_variable('Biases', [output_dim], initializer=tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(input_, w, strides=[1, stride[0], stride[1], 1], padding='SAME')
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv


# decompose convolution
def deconv2d(input_, output_shape,kernal=[5,5], stride=[2,2], stride_w=2, stddev=0.02, name=None, with_w=False):
    with tf.variable_scope(name):

        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('Weight', [kernal[0], kernal[1], output_shape[-1], input_.get_shape()[-1]],initializer=norm_init(stddev=stddev))
        biases = tf.get_variable('Biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, stride[0], stride[1], 1])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


# batch norm
def batch_norm(x, epsilon=1e-5, decay=0.9, name=None, is_train=False, scale=False):
    return tf.contrib.layers.batch_norm(x, is_training=is_train, epsilon=epsilon, decay = decay,  updates_collections=None, scope=name, scale=scale)

# Random position 
def rand_noise(batch_size, dim):
    return np.random.uniform(-1.0, 1.0, size=[batch_size, dim]).astype(np.float32)


def cons_init(value):
    return tf.constant_initializer(value=value)


def norm_init(stddev=0.02):
    return tf.truncated_normal_initializer(stddev=stddev)




