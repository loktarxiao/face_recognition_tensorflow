#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:23:42 2018

@author: loktarxiao
"""

import tensorflow as tf

def _conv2d(input_tensor, ksize, num_filter, name):
    initializer = tf.contrib.layers.xavier_initializer()
    regularizer = tf.nn.l2_loss
    activition = tf.nn.leaky_relu
    activity_regularizer = tf.layers.batch_normalization

    output = tf.layers.conv2d(input_tensor, num_filter, ksize,
                padding = 'SAME',
                activation = activition,
                activity_regularizer = activity_regularizer,
                kernel_initializer = initializer,
                kernel_regularizer = regularizer,
                name = name)
    return output

def _maxpool2d(input_tensor, pool_size, stride, name):
    output = tf.layers.max_pooling2d(input_tensor, pool_size,
                    strides = stride, name=name)
    return output

def _conv_block(input_tensor, num_filter, block_name, reuse=None):
    with tf.variable_scope(block_name, reuse=reuse):
        conv1 = _conv2d(input_tensor, 3, num_filter, 'conv1')
        conv2 = _conv2d(conv1, 1, num_filter, 'conv2')
        maxpool = _maxpool2d(conv2, 2, 2, 'maxpool')
    return maxpool

def alexnet(input_tensor, reuse=None):
    with tf.variable_scope("alexnet", reuse=reuse):
        block1 = _conv_block(input_tensor, 32, "conv_block1")
        block2 = _conv_block(block1, 64, "conv_block2")
        block3 = _conv_block(block2, 128, "conv_block3")
    
    return block3

if __name__ == '__main__':

    input_1 = tf.random_normal((1,224,224,3))
    input_2 = tf.ones((1,224,224,3))
    output1 = alexnet(input_1)
    output2 = alexnet(input_1, reuse=True)

    output = tf.abs(output1 - output2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    out = sess.run([output])
    train_writer = tf.summary.FileWriter('test', sess.graph)
    train_writer.close()
    sess.close()
    print(out[0].shape)
    import matplotlib.pyplot as plt
    plt.imshow(out[0][0,:,:,:3])
    plt.show()
    