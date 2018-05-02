#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:23:42 2018

@author: loktarxiao
"""

import tensorflow as tf
from model_zoo import *



class MODEL(object):
    
    def __init__(self, input_shape, num_channel):
        self.input_shape = input_shape
        self.num_channel = num_channel
        self.image_shape = [None]+self.input_shape+[self.num_channel]

    def build_model(self):
        with tf.variable_scope("input_block"):
            image1 = tf.placeholder(tf.float32, 
                shape=self.image_shape, name = 'image1')
            
            image2 = tf.placeholder(tf.float32, 
                shape=self.image_shape, name = 'image2')

            label = tf.placeholder(tf.float32, shape=None, name='label')

        with tf.variable_scope("control_params"):
            keep_prob = tf.placeholder_with_default(1, shape=(1,), name='keep_prob_ratio')
            init = tf.global_variables_initializer()

        with tf.variable_scope("feature"):
            feature1 = alexnet(image1)
            feature2 = alexnet(image2, reuse=True)

            feature = tf.abs(feature1 - feature2)