import argparse
import os

import model
import util

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Classify FASHION MNIST images.')
parser.add_argument(
    '--data_dir',
    type=str,
    default='/work/cse496dl/shared/homework/01/',
    help='directory where FMNIST is located')
parser.add_argument(
    '--model_dir',
    type=str,
    default='./homework1_logs',
    help='directory where model graph and weights are saved')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--earlystop', type=int, default=5)
parser.add_argument('--regularizer', action='store_true')
parser.add_argument('--use_learning_rate_2', action='store_true')
parser.add_argument('--use_model_2', action='store_true')
args = parser.parse_args()

# load data
train_images = np.load(os.path.join(args.data_dir, 'fmnist_train_data.npy'))
train_labelNums = np.load(os.path.join(args.data_dir, 'fmnist_train_labels.npy'))
train_labels = util.convertLabelsToOneHot(train_labelNums)

# specify the network
x = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
tf.identity(x, name='input_placeholder')

if args.use_model_2:
    output = model.createModelTwo(x)
else:
    output = model.createModelOne(x)

# define classification loss
y = tf.placeholder(tf.float32, [None, 10], name='label')
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=output)
regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# this is the weight of the regularization part of the final loss
REG_COEFF = 0.1
# this value is what we'll pass to `minimize`
total_loss = cross_entropy
if args.regularizer:
    total_loss = cross_entropy + REG_COEFF * sum(regularization_losses)

total_loss_op = tf.reduce_mean(total_loss)

confusion_matrix_op = tf.confusion_matrix(
    tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)

# set up training and saving functionality
global_step_tensor = tf.get_variable(
    'global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)

if args.use_learning_rate_2:
    learningRate = 0.0001
else:
    learningRate = 0.001

optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
saver = tf.train.Saver()

# Get Test set
X_trainingSet, X_test, y_trainingSet, y_test = train_test_split(train_images, train_labels, test_size=0.20)

# splitting into 90% and 10%
X_train, X_val, y_train, y_val = train_test_split(
         X_trainingSet, y_trainingSet, test_size=0.10)

train_num_examples = X_train.shape[0]
val_num_examples = X_val.shape[0]
test_num_examples = X_test.shape[0]

print(train_num_examples)
print(val_num_examples)
print(test_num_examples)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # run training
    batch_size = args.batch_size
    
    ce_vals = []
    conf_mxs = []
    for i in range(val_num_examples // batch_size):
        batch_xs = X_val[i * batch_size:(i + 1) * batch_size, :]
        batch_ys = y_val[i * batch_size:(i + 1) * batch_size, :]
        val_ce, conf_matrix = session.run(
            [total_loss_op, confusion_matrix_op], {
                x: batch_xs,
                y: batch_ys
            })
        ce_vals.append(val_ce)
        conf_mxs.append(conf_matrix)
    store = sum(ce_vals) / len(ce_vals)

    for epoch in range(args.epochs):
        print('Epoch: ' + str(epoch))

        # run gradient steps and report mean loss on train data
        ce_vals = []
        for i in range(train_num_examples // batch_size):
            batch_xs = X_train[i * batch_size:(i + 1) * batch_size, :]
            batch_ys = y_train[i * batch_size:(i + 1) * batch_size, :]
            _, train_ce = session.run(
                [train_op, total_loss_op], {
                    x: batch_xs,
                    y: batch_ys
                })
            ce_vals.append(train_ce)
        avg_train_ce = sum(ce_vals) / len(ce_vals)
        print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
        
        
         # report mean val loss
        ce_vals = []
        conf_mxs = []
        for i in range(val_num_examples // batch_size):
            batch_xs = X_val[i * batch_size:(i + 1) * batch_size, :]
            batch_ys = y_val[i * batch_size:(i + 1) * batch_size, :]
            val_ce, conf_matrix = session.run(
                [total_loss_op, confusion_matrix_op], {
                    x: batch_xs,
                    y: batch_ys
                })
            ce_vals.append(val_ce)
            conf_mxs.append(conf_matrix)
        avg_val_ce = sum(ce_vals) / len(ce_vals)
        print('VALIDATION CROSS ENTROPY: ' + str(avg_val_ce))
        print('VALIDATION CONFUSION MATRIX:')
        print(str(sum(conf_mxs)))
        
        if store < avg_val_ce:
            counter=counter+1
            if counter==args.earlystop:              
                print("STOPPING EPOCH",epoch)
                break
        else:
            path_prefix = saver.save(session, os.path.join(args.model_dir, "homework_1"), global_step=global_step_tensor)
            counter=0
            store=avg_val_ce

        # report mean test loss
        ce_vals = []
        conf_mxs = []
        for i in range(test_num_examples // batch_size):
            batch_xs = X_test[i * batch_size:(i + 1) * batch_size, :]
            batch_ys = y_test[i * batch_size:(i + 1) * batch_size, :]
            test_ce, conf_matrix = session.run(
                [total_loss_op, confusion_matrix_op], {
                    x: batch_xs,
                    y: batch_ys
                })
            ce_vals.append(test_ce)
            conf_mxs.append(conf_matrix)
        avg_test_ce = sum(ce_vals) / len(ce_vals)
        print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
        print('TEST CONFUSION MATRIX:')
        print(str(sum(conf_mxs)))


