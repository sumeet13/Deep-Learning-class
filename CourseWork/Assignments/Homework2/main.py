import argparse
import os
import csv
import model
import util
import datetime
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Classify FASHION MNIST images.')
parser.add_argument(
    '--data_dir',
    type=str,
    default='/work/cse496dl/shared/homework/02',
    help='directory where FMNIST is located')
parser.add_argument(
    '--model_dir',
    type=str,
    default='./homework2_logs',
    help='directory where model graph and weights are saved')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--earlystop', type=int, default=50)
parser.add_argument('--autoencoder_epochs', type=int, default=100)
parser.add_argument('--autoencoder_earlystop', type=int, default=5)
parser.add_argument('--regularizer', action='store_true')
parser.add_argument('--use_learning_rate_2', action='store_true')
parser.add_argument('--use_model_2', action='store_true')
parser.add_argument('--autoencoder', action='store_true')
parser.add_argument('--model_type', type=int, default=2)
args = parser.parse_args()

# load data
train_imagesFlat = np.load(os.path.join(args.data_dir, 'cifar_images.npy'))
train_images = np.reshape(train_imagesFlat, [-1, 32, 32, 3])
train_labelNums = np.load(os.path.join(args.data_dir, 'cifar_labels.npy'))
train_labels = util.convertLabelsToOneHot(train_labelNums)

imagenet_imagesFlat = np.load(os.path.join(args.data_dir, 'imagenet_images.npy'))
imagenet_images = np.reshape(imagenet_imagesFlat, [-1, 32, 32, 3])

# specify the network
x = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_placeholder')
tf.identity(x, name='input_placeholder')
filename = os.path.join(args.model_dir, "results.csv")
print("filename", filename)
'''
if args.autoencoder == False:
    if args.use_model_2:
        output = model.createCNNModelTwo(x)
    else:
        output = model.createCNNModelOne(x)
else:
    autoencoderLayers = model.createAutoencoderModelOne(x)
    autoencoderOutput = autoencoderLayers[0]
    output = autoencoderLayers[1]'''
 
if args.model_type== 1:
    output = model.createCNNModelOne(x)
if args.model_type== 2:
    output = model.createCNNModelTwo(x)
if args.model_type== 3:
    output = model.createCNNModelThree(x)
if args.model_type== 4:
    output = model.createCNNModelFour(x)
if args.model_type== 5:
    autoencoderLayers = model.createAutoencoderModelOne(x)
    autoencoderOutput = autoencoderLayers[0]
    output = autoencoderLayers[1]


print(tf.shape(output))
# define classification loss
y = tf.placeholder(tf.float32, [None, 100], name='label')
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output)
regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, "linear_model")
# this is the weight of the regularization part of the final loss
REG_COEFF = 0.1
# this value is what we'll pass to `minimize`
total_loss = cross_entropy
if args.regularizer:
    total_loss = cross_entropy + REG_COEFF * sum(regularization_losses)

total_loss_op = tf.reduce_mean(total_loss)

confusion_matrix_op = tf.confusion_matrix(
    tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=100)

# set up training and saving functionality
global_step_tensor = tf.get_variable(
    'global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)

if args.use_learning_rate_2:
    learningRate = 0.00003
else:
    learningRate = 0.001

# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
train_op = optimizer.minimize(total_loss, global_step=global_step_tensor, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "linear_model"))

# Autoencoder optimizer
if args.autoencoder:
    # Autoencoder Loss
    autoencoder_cross_entropy = tf.reduce_mean(tf.square(autoencoderOutput - x))
    autoencoder_regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, "autoencoder")
    autoencoder_total_loss = autoencoder_cross_entropy
    if args.regularizer:
        autoencoder_total_loss = autoencoder_total_loss + REG_COEFF * sum(autoencoder_regularization_losses)
    
    autoencoder_total_loss_op = tf.reduce_mean(autoencoder_total_loss
    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
    autoencoder_train_op = autoencoder_optimizer.minimize(autoencoder_total_loss, global_step=global_step_tensor, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "autoencoder"))

saver = tf.train.Saver()

# Get Test set
X_trainingSet, X_test, y_trainingSet, y_test = train_test_split(train_images, train_labels, test_size=0.05)

# splitting into 90% and 10%
X_train, X_val, y_train, y_val = train_test_split(
         X_trainingSet, y_trainingSet, test_size=0.10)

train_num_examples = X_train.shape[0]
val_num_examples = X_val.shape[0]
test_num_examples = X_test.shape[0]
print('Training Num ' + str(train_num_examples))
print('Validation Num ' + str(val_num_examples))
print('Test Num ' + str(test_num_examples))

# Get Autoencoder Test set
X_autoencoder_trainingSet, X_autoencoder_test = train_test_split(imagenet_images, test_size=0.05)

# splitting autoencoder into 90% and 10%
X_autoencoder_train, X_autoencoder_val = train_test_split(X_autoencoder_trainingSet, test_size=0.10)

train_autoencoder_num_examples = X_autoencoder_train.shape[0]
val_autoencoder_num_examples = X_autoencoder_val.shape[0]
test_autoencoder_num_examples = X_autoencoder_test.shape[0]
print('Training Num ' + str(train_autoencoder_num_examples))
print('Validation Num ' + str(val_autoencoder_num_examples))
print('Test Num ' + str(test_autoencoder_num_examples))

util.createCSV(filename)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    # run training
    batch_size = args.batch_size
    stop=False
    if args.autoencoder:
        for epoch in range(args.autoencoder_epochs):

            ae_vals = []
            for i in range(val_autoencoder_num_examples // batch_size):
                batch_xs = X_autoencoder_val[i * batch_size:(i + 1) * batch_size, :]
                ae_loss = session.run(
                    autoencoder_total_loss_op, {
                        x: batch_xs
                    })
                ae_vals.append(ae_loss)
            store = sum(ae_vals) / len(ae_vals)
            counter = 0

            print('Autoencoder Epoch: ' + str(epoch))
            # run gradient steps and report mean loss on train data
            ae_vals = []
            for i in range(train_autoencoder_num_examples // batch_size):
                batch_xs = X_autoencoder_train[i * batch_size:(i + 1) * batch_size, :]      
                _, ae_train_ce = session.run(
                    [autoencoder_train_op, autoencoder_total_loss_op], {
                        x: batch_xs
                    })
                ae_vals.append(ae_train_ce)
            avg_ae_train_ce = sum(ae_vals) / len(ae_vals)
            print('TRAIN CROSS ENTROPY: ' + str(avg_ae_train_ce))
            
            
            # report mean val loss
            ae_vals = []
            for i in range(val_autoencoder_num_examples // batch_size):
                batch_xs = X_autoencoder_val[i * batch_size:(i + 1) * batch_size, :]
                val_ae_loss = session.run(
                    autoencoder_total_loss, {
                        x: batch_xs
                    })
                ae_vals.append(val_ae_loss)
            avg_val_ae = sum(ae_vals) / len(ae_vals)
            print('VALIDATION CROSS ENTROPY: ' + str(avg_val_ae))
            
            if store > avg_val_ae:
                counter=counter+1
                if counter==args.autoencoder_earlystop:              
                    print("STOPPING EPOCH", epoch)
                    stop=True
            else:
                path_prefix = saver.save(session, os.path.join(args.model_dir, "homework_2"), global_step=global_step_tensor)
                counter=0
                store=avg_val_ae

            # report mean test loss
            
            ae_vals = []
            for i in range(test_autoencoder_num_examples // batch_size):
                batch_xs = X_autoencoder_test[i * batch_size:(i + 1) * batch_size, :]
                test_ae = session.run(
                    autoencoder_total_loss, {
                        x: batch_xs
                    })
                ae_vals.append(test_ae)
            avg_test_ae = sum(ae_vals) / len(ae_vals)
            print('AUTOENCODER TEST CROSS ENTROPY: ' + str(avg_test_ae))
            #print('AUTOENCODER TEST ACCURACY: ' + str(test_accuracy_ae))
            
            lines=[str(args.model_type), str(test_autoencoder_num_examples),str(batch_size), str(args.autoencoder_epochs), str(args.autoencoder_earlystop),      str('-'), str(avg_test_ae),str('-'),
               str('-'), str(avg_val_ae),str('-'),
               str(avg_ae_train_ce),str(learningRate)]
        
            print(lines)
            util.appendCSV(filename, lines)
            if stop == True:              
                break
    
    util.appendCSV(filename,"\n")        
    lines=['---', '---','---', '---', '---',     '---', '---','---',
               '---','---','---','---',
               '---','---']  
    util.appendCSV(filename,lines)
    util.appendCSV(filename,"\n")
    print("args.autoencoder",args.autoencoder)
    ce_vals = []
    conf_mxs = []
    for i in range(val_num_examples // batch_size):
        batch_xs = X_val[i * batch_size:(i + 1) * batch_size, :]
        batch_ys = y_val[i * batch_size:(i + 1) * batch_size, :]
        ce_val = session.run(
            total_loss_op, {
                x: batch_xs,
                y: batch_ys
            })
        ce_vals.append(ce_val)
    store = sum(ce_vals) / len(ce_vals)
    print('Validation Loss: ' + str(store))
    counter = 0
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
        val_correct = sum(sum(conf_mxs)[i][i] for i in range(100))
        print('VALIDATION CORRECT: ' + str(val_correct))
        val_accuracy = val_correct / val_num_examples
        print('VALIDATION ACCURACY: '+ str(val_accuracy))
        if store < avg_val_ce:
            counter=counter+1
            if counter==args.earlystop:              
                print("STOPPING EPOCH",epoch)
                """break"""
                stop = True
        else:
            path_prefix = saver.save(session, os.path.join(args.model_dir, "homework_2"), global_step=global_step_tensor)
            counter=0
            store=avg_val_ce
            conf_mx = sum(conf_mxs)
            util.writeConfusionMatrix(os.path.join(args.model_dir, "validation_confusion_matrix.txt"), conf_mx)
            

        
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
        test_correct = sum(sum(conf_mxs)[i][i] for i in range(100))
        test_accuracy = test_correct / test_num_examples
        print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
        print('TEST CORRECT: ' + str(test_correct)) 
        print('TEST ACCURACY: ' + str(test_accuracy)) 
        
        lines=[str(args.model_type), str(test_num_examples),str(batch_size), str(args.epochs), str(args.earlystop),      str(test_accuracy), str(avg_test_ce),str(test_correct),
               str(val_accuracy), str(avg_val_ce),str(val_correct),
               str(avg_train_ce),str(learningRate)]
        
        print(lines)
        util.appendCSV(filename,lines)
        if stop== True:
            break
        
