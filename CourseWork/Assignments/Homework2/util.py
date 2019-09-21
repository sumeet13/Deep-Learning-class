import argparse
import os

import numpy as np
import tensorflow as tf
import csv

def convertLabelsToOneHot(labels):
    oneHotEncoded = []
    indexLabels = labels.astype(np.int64)
    for label in indexLabels:
        row = [0] * 100
        row[label] = 1
        oneHotEncoded.append(row)
    return np.array(oneHotEncoded)

def createCSV(filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
        
    with open(filename, mode='w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['Model type', 'test_examples','batch size', 'epoch', 'early stop epoch', 'test accuracy','test_cross_entropy','test_correct','val accuracy','val_cross_entropy','val_correct','train_cross_entropy','learning_rate'])
        
def appendCSV(filename,lines):
    with open(filename, 'a') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(lines)
            
def writeConfusionMatrix(filename,con_mat):
    with open(filename, mode='w') as csvfile:
        for i in range(100):
            for j in range(99):
                csvfile.write(str(con_mat[i][j]) + ',')
            csvfile.write(str(con_mat[i][99]) + '\n')