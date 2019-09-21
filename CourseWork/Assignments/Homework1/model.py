import argparse
import os

import numpy as np
import tensorflow as tf

def createModelOne(x):
    with tf.name_scope('linear_model') as scope:
        x = x / 255.0
        hidden = tf.layers.dense(x,
                                400,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                activation=tf.nn.relu, 
                                name='hidden_layer')
        output = tf.layers.dense(hidden,
                                10,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                name='output_layer')
    tf.identity(output, name='output')
    return output

def createModelTwo(x):
    with tf.name_scope('linear_model') as scope:
        x = x / 255.0
        hidden1 = tf.layers.dense(x,
                                20,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                activation=tf.nn.relu, 
                                name='hidden_layer_1')
        hidden2 = tf.layers.dense(hidden1,
                                20,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                activation=tf.nn.relu, 
                                name='hidden_layer_2')
        output = tf.layers.dense(hidden2,
                                10,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                name='output_layer')
    tf.identity(output, name='output')
    return output
