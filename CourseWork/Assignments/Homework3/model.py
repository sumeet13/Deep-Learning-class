import numpy as np
import tensorflow as tf

def dqn_network(input_tensor, input_shape, action_num, variable_scope):
    with tf.variable_scope(variable_scope):
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_tensor=input_tensor),
            tf.keras.layers.Conv2D(16, 5, 2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(32, 5, 2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(32, 5, 2, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(action_num, activation="softmax")
        ])

def dqn_network_2(input_tensor, input_shape, action_num, variable_scope):
    with tf.variable_scope(variable_scope):
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_tensor=input_tensor),
            tf.keras.layers.Conv2D(16, 5, 2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(32, 5, 2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, 5, 2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(128, 5, 2, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(action_num, activation="softmax")
        ])

def dqn_network_3(input_tensor, input_shape, action_num, variable_scope):
    with tf.variable_scope(variable_scope):
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_tensor=input_tensor),
            tf.keras.layers.Conv2D(16, 5, 2, padding="same", activation="relu", name="conv_layer_1"),
            tf.keras.layers.Conv2D(32, 5, 2, padding="same", activation="relu", name="conv_layer_2"),
            tf.keras.layers.Conv2D(32, 5, 2, padding="same", activation="relu", name="conv_layer_3"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu", name="dense_1"),
            tf.keras.layers.Dense(action_num, activation="softmax", name="dense_2")
        ])

def dqn_network_4(input_tensor, input_shape, action_num, variable_scope):
    with tf.variable_scope(variable_scope):
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_tensor=input_tensor),
            tf.keras.layers.Conv2D(32, 8, 4, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, 4, 2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, 3, 1, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(action_num, activation="softmax")
        ])

def dqn_network_5(input_tensor, input_shape, action_num, variable_scope):
    with tf.variable_scope(variable_scope):
        return tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_tensor=input_tensor),
            tf.keras.layers.Conv2D(16, 5, 2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(32, 5, 2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(64, 5, 2, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(128, 5, 2, padding="same", activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(action_num, activation="softmax")
        ])
