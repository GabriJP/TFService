# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

from sys import argv, stderr

from DataSet import DataSet
from nn import nn

import Inception

"""
Program usage:
python ProgramName.py classes=directory resize='width'x'height' crop=x_from:y_from:x_to:y_to
train=percentage_for_training val=percentage_for_validation test=percentage_for_validation out=output_file

Example:
python Main.py classes=Other/Classes/Carreteras resize=140x80 crop=0:0:140:80 train=0.8 val=0 test=0.2 out=Other/Output/
"""

classes_root = [arg[8:] for arg in argv if arg.startswith("classes=")]
if len(classes_root) < 1:
    print("Not enough class arguments", file=stderr)
    exit(1)
classes_root = classes_root[0]

resize = [arg[7:] for arg in argv if arg.startswith("resize=")]
if len(resize) < 1:
    print("Not enough resize arguments", file=stderr)
    exit(1)
resize = tuple(map(int, resize[0].split("x")))

crop = [arg[5:] for arg in argv if arg.startswith("crop=")]
if len(crop) < 1:
    print("Not enough crop arguments", file=stderr)
    exit(1)
crop = tuple(map(int, crop[0].split(":")))

train = [arg[6:] for arg in argv if arg.startswith("train=")]
if len(train) < 1:
    print("Not enough train arguments", file=stderr)
    exit(1)
train = float(train[0])

test = [arg[5:] for arg in argv if arg.startswith("test=")]
if len(test) < 1:
    print("Not enough test arguments", file=stderr)
    exit(1)
test = float(test[0])

validation = [arg[4:] for arg in argv if arg.startswith("val=")]
if len(validation) < 1:
    print("Not enough val arguments", file=stderr)
    exit(1)
validation = float(validation[0])

output = [arg[4:] for arg in argv if arg.startswith("out=")]
if len(output) < 1:
    print("Not enough out arguments", file=stderr)
    exit(1)
output = output[0]

data_set = DataSet.from_directory(classes_root, resize, crop, train, test)
data_set.to_file(output)
data_set = DataSet.from_file(output)
# labels, frames = data_set.next_test_batch(10)

# from matplotlib import cm
# from matplotlib import pyplot as plt
# for p in range(10):
#     plt.subplot(2, 5, p + 1)
#     plt.imshow(frames[p], cmap=cm.Greys_r)
#     plt.xlabel(labels[p])
# plt.show()

Inception.nn(data_set)
