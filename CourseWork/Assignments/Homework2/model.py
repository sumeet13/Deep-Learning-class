import argparse
import os

import numpy as np
import tensorflow as tf

def createCNNModelOne(x):
    with tf.variable_scope('linear_model') as scope:
        hidden_1 = tf.layers.conv2d(x, 32, 5, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu, name='hidden_1')
        pool_1 = tf.layers.max_pooling2d(hidden_1, 2, 2, padding='same', name='pool_1')
        
        hidden_2 = tf.layers.conv2d(pool_1, 64, 5, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu, name='hidden_2')
        pool_2 = tf.layers.max_pooling2d(hidden_2, 2, 2, padding='same', name='pool_2')
        # followed by a dense layer output 

        flat = tf.reshape(pool_2, [-1, 8*8*64]) # flatten from 4D to 2D for dense layer
        hidden_dense_1 = tf.layers.dense(flat, 800, activation=tf.nn.relu, name='hidden_dense_1')
        output = tf.layers.dense(hidden_dense_1, 100, name='output')

    tf.identity(output, name='output')
    return output

def createCNNModelTwo(x):
    with tf.variable_scope('linear_model') as scope:
        hidden_1 = tf.layers.conv2d(x, 64, 3, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='hidden_1')
        hidden_2 = tf.layers.conv2d(hidden_1, 128, 3, padding='same', 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='hidden_2')
        pool_2 = tf.layers.max_pooling2d(hidden_2, 2, 2, padding='same', name='pool_2')
        
        hidden_3 = tf.layers.conv2d(pool_2, 256, 3, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='hidden_3')
        hidden_4 = tf.layers.conv2d(hidden_3, 512, 3, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu, name='hidden_4')
        pool_4 = tf.layers.max_pooling2d(hidden_4, 2, 2, padding='same', name='pool_4')
        # followed by a dense layer output 

        flat = tf.reshape(pool_4, [-1, 8*8*512]) # flatten from 4D to 2D for dense layer
        output = tf.layers.dense(flat, 100, name='output')

    tf.identity(output, name='output')
    return output

def createAutoencoderModelOne(x):
    with tf.variable_scope('autoencoder') as scope:
        hidden_1 = tf.layers.conv2d(x, 64, 3, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='hidden_1')
        hidden_2 = tf.layers.conv2d(hidden_1, 128, 3, padding='same', 
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='hidden_2')
        pool_2 = tf.layers.max_pooling2d(hidden_2, 2, 2, padding='same', name='pool_2')
        
        hidden_3 = tf.layers.conv2d(pool_2, 256, 3, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='hidden_3')
        hidden_4 = tf.layers.conv2d(hidden_3, 512, 3, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='hidden_4')
        pool_4 = tf.layers.max_pooling2d(hidden_4, 2, 2, padding='same', name='pool_4')
        # followed by a dense layer output 

        flat = tf.reshape(pool_4, [-1, 8*8*512]) # flatten from 4D to 2D for dense layer
        code = tf.layers.dense(flat, 400,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='code')

        # Decoder
        hidden_decoder = tf.layers.dense(code, 8*8*512,
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='hidden_decoder')
        hidden_decoder_reshape = tf.reshape(hidden_decoder, [-1, 8, 8, 512])
        upsample_4 = tf.layers.conv2d_transpose(hidden_decoder_reshape, 512, 2, strides=2, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='upsample_4')
        decoder_4 = tf.layers.conv2d_transpose(upsample_4, 256, 3, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='decoder_4')
        decoder_3 = tf.layers.conv2d_transpose(decoder_4, 128, 3, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='decoder_3')
        upsample_2 = tf.layers.conv2d_transpose(decoder_3, 128, 2, strides=2, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='upsample_2')
        decoder_2 = tf.layers.conv2d_transpose(upsample_2, 64, 3, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='decoder_2')
        autoencoder_output = tf.layers.conv2d_transpose(decoder_2, 3, 3, padding='same',
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                    activation=tf.nn.relu,
                                    name='autoencoder_output')
        
    with tf.variable_scope('linear_model') as scope:
        hidden_1 = tf.layers.dense(code, 400, activation=tf.nn.relu, name='hidden_1')
        output = tf.layers.dense(hidden_1, 100, name='output')

    tf.identity(output, name='output')
    return [autoencoder_output, output]