#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:23:42 2018

@author: loktarxiao
"""

import tensorflow as tf

def _image_augmentation(image_tensor, output_size=(224, 224, 3)):
    #image = tf.image.resize_area(image_tensor, (250, 250))
    image = tf.image.random_flip_up_down(image_tensor)
    image = tf.random_crop(image, output_size)
    return image


def _label_augmentation(label_tensor, class_num):
    label = tf.one_hot(label_tensor, class_num)
    return label


def slice_process(image1, image2, label):
    image1 = _image_augmentation(image1)
    image2 = _image_augmentation(image2)
    label = _label_augmentation(label, 2)
    return image1, image2, label

def parse_function(example_proto):
    dics = {
        'image1':tf.FixedLenFeature(shape=(), dtype=tf.string),
        'image2':tf.FixedLenFeature(shape=(), dtype=tf.string),
        'image1_shape':tf.FixedLenFeature(shape=(3, ), dtype=tf.int64),
        'image2_shape':tf.FixedLenFeature(shape=(3, ), dtype=tf.int64),
        'label':tf.FixedLenFeature(shape=(), dtype=tf.int64)
    }
    parsed_example = tf.parse_single_example(example_proto, dics)

    image1 = tf.reshape(tf.decode_raw(parsed_example['image1'], tf.uint8), parsed_example['image1_shape'])
    image2 = tf.reshape(tf.decode_raw(parsed_example['image2'], tf.uint8), parsed_example['image1_shape'])
    label = parsed_example['label']

    return image1, image2, label

if __name__ == '__main__':
    filenames = ['../data/raw_data/1st/train.tfrecord']
    dataset = tf.data.TFRecordDataset(filenames)
    new_dataset = dataset.map(parse_function, num_parallel_calls=8)
    new_dataset = new_dataset.shuffle(buffer_size=10000).repeat(8)
    new_dataset = new_dataset.map(slice_process, num_parallel_calls=8)
    new_dataset = new_dataset.batch(4)
    iterator = new_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    sess = tf.InteractiveSession()

    for i in range(40):
        try:
            image1, image2, label = sess.run([next_element])[0]
            print(label)
            print(image1.shape)
        except tf.errors.OutOfRangeError:
            break