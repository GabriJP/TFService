# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

from sys import stderr
from DataSet import DataSet
from CNN import create_cnn, play_cnn

import argparse

"""
Example:
-r 0.75 -e 0.25 -i 3000 -c Other/Classes/Carreteras -p Other/Classes/Carreteras/tunel/tunel.mp4 140x80 0:0:140:80 Other/Output/
"""

parser = argparse.ArgumentParser(prog="TFService", description="Process videos and train a CNN with their frames")
parser.add_argument("resize", help="Resize frames to this. 140x80.")
parser.add_argument("crop", help="Crop resized frames to this. 0:0:140:80 (Left:Up:Right:Bottom).")
parser.add_argument("output", help="Directory where to save the dataset and the network data.")
parser.add_argument("-r", "--train_p", type=float, help="Percentage of frames to use in the training process. 0.8.",
                    default=0.8)
parser.add_argument("-e", "--test_p", type=float, help="Percentage of frames to use in the testing process. 0.2.",
                    default=0.2)
parser.add_argument("-i", "--iterations", type=int,
                    help="Number of frames presented to the net in the training process. 3000.", default=3000)
parser.add_argument("-c", "--create", help="Directory where to find the directories of classes with videos.")
parser.add_argument("-p", "--play", help="Use this video to play the loaded CNN")
args = parser.parse_args()

classes_root = args.create
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

if classes_root:
    data_set = DataSet.from_directory(classes_root, resize, crop, train, test)
    data_set.to_file(output)
    create_cnn(data_set, output, iterations)

if args.play:
    data_set = DataSet.from_file(output)
    play_cnn(data_set.get_metadata(), output, args.play)
