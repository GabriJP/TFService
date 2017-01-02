# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

from sys import stderr
from DataSet import DataSet
from CNNCreator import create_cnn
from CNNPlayer import play_cnn

import argparse
import tensorflow as tf

"""
Program usage:
python ProgramName.py classes=directory resize='width'x'height' crop=x_from:y_from:x_to:y_to
train=percentage_for_training val=percentage_for_validation test=percentage_for_validation out=output_file

Example:
-r 0.75 -e 0.25 -i 3000 Other/Classes/Carreteras 140x80 0:0:140:80 Other/Output/
"""

parser = argparse.ArgumentParser()
parser.add_argument("directory", help="Directory where to find the directories of classes with videos.")
parser.add_argument("resize", help="Resize frames to this. 140x80.")
parser.add_argument("crop", help="Crop resized frames to this. 0:0:140:80 (Left:Up:Right:Bottom).")
parser.add_argument("output", help="Directory where to save the dataset and the network data.")
parser.add_argument("-r", "--train_p", type=float, help="Percentage of frames to use in the training process. 0.8.",
                    default=0.8)
parser.add_argument("-e", "--test_p", type=float, help="Percentage of frames to use in the testing process. 0.2.",
                    default=0.2)
parser.add_argument("-i", "--iterations", type=int,
                    help="Number of frames presented to the net in the training process. 3000.", default=3000)
args = parser.parse_args()

classes_root = args.directory
resize = tuple(map(int, args.resize.split("x")))
crop = tuple(map(int, args.crop.split(":")))
output = args.output
train = args.train_p
if not 0 <= train <= 1:
    print("Train percentage must be between 0 and 1.", file=stderr)
    exit(1)
test = args.test_p
if not 0 <= test <= 1:
    print("Test percentage must be between 0 and 1.", file=stderr)
    exit(1)
if not 0 <= test + train <= 1:
    print("Test + train percentages must be between 0 and 1.", file=stderr)
    exit(1)
validation = 1 - train - test
iterations = args.iterations

# data_set = DataSet.from_directory(classes_root, resize, crop, train, test)
# data_set.to_file(output)
data_set = DataSet.from_file(output)

# create_cnn(data_set, output, iterations)
# tf.reset_default_graph()
play_cnn(data_set.get_metadata(), output)
