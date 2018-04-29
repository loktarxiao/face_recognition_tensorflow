#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:23:42 2018

@author: loktarxiao
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
import os
import random


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def get_image_binary(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def tfrecord_string(image1, image2, label):
    feature = {
        'label': _int64_feature([label]),
        'image1': _bytes_feature(image1.tostring()),
        'image2': _bytes_feature(image2.tostring()),
        'image1_shape': _int64_feature(image1.shape),
        'image2_shape': _int64_feature(image2.shape)
    }
    tf_features = tf.train.Features(feature=feature)
    example = tf.train.Example(features=tf_features)

    tf_serialized = example.SerializeToString()

    return tf_serialized


def main(dataset_path, train_ratio=0.7):
    datalist_path = os.path.join(dataset_path, 'datalist.txt')
    train_writer = tf.python_io.TFRecordWriter(os.path.join(dataset_path, 'train.tfrecord'))
    valid_writer = tf.python_io.TFRecordWriter(os.path.join(dataset_path, 'valid.tfrecord'))

    with open(datalist_path, 'r') as fp:
        lines = fp.readlines()
    random.shuffle(lines)
    edge = len(lines)*0.7
    
    for i in tqdm(range(len(lines))):
        mess = lines[i].strip().split(':')
        image1_path = os.path.join(dataset_path, mess[0])
        image2_path = os.path.join(dataset_path, mess[1])
        label = int(mess[2])
        
        image1 = get_image_binary(image1_path)
        image2 = get_image_binary(image2_path)

        tf_serialized = tfrecord_string(image1, image2, label)

        if i < edge:
            train_writer.write(tf_serialized)
        else:
            valid_writer.write(tf_serialized)
    
    train_writer.close()
    valid_writer.close()


if __name__ == '__main__':
    dataset_path = '../data/raw_data/1st'
    main(dataset_path)
