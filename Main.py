# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
from GUI import *
from Video import Video
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pprint import pprint

from CódigoEjemplo.serialization import get_pickle_dictionary

# get_pickle_dictionary("/home/pe/wallpapers")

n = getclassnum()
clases = []

for i in range(1, n + 1):
    clases.append([getclassname(i), getclassdirectory(i)])

redimensions = getredimension()
crop = getcrop()
percentages = getpercentages()
output = getoutputdrectory()

frames = []
for clas in clases:
    for (dirpath, dirnames, filenames) in os.walk(clas[1]):
        for file in filenames:
            for frame in Video(os.path.join(dirpath, file)):
                frames.append((clas[0], frame))

items_for_training = int(percentages / 100 * len(frames))
np.random.shuffle(frames)

data_matrix_train = np.array([channel for channel in [np.array(t[1]) for t in frames[:items_for_training]]])
label_list_train = np.array([t[0] for t in frames[:items_for_training]])
data_matrix_test = np.array([channel for channel in [t[1].flatten() for t in frames[items_for_training:]]])
label_list_test = np.array([t[0] for t in frames[items_for_training:]])

train = {"data": data_matrix_train.astype("int"), "labels": label_list_train.astype("int")}
test = {"data": data_matrix_test.astype("int"), "labels": label_list_test.astype("int")}

pickle.dump(train, open(os.path.join(output, "data_objects"), "wb"))
pickle.dump(test, open(os.path.join(output, "test_objects"), "wb"))
