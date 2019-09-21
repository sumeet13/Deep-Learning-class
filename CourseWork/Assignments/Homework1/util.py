import argparse
import os

import numpy as np
import tensorflow as tf


def convertLabelsToOneHot(labels):
    oneHotEncoded = []
    indexLabels = labels.astype(np.int64)
    for label in indexLabels:
        row = [0] * 10
        row[label] = 1
        oneHotEncoded.append(row)
    return np.array(oneHotEncoded)