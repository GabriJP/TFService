# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

from sys import stderr
from DataSet import DataSet
from CNN import create_cnn, play_cnn

import argparse

"""
Example:
-d Other/Classes/Carreteras 140x80 0:0:140:80 0.75 0.25 -n 2000 -p Other/Classes/Carreteras/tunel/tunel.mp4 Other/Output
"""

parser = argparse.ArgumentParser(prog="TFService", description="Process videos and train a CNN with their frames")
parser.add_argument("-d", "--dataset", nargs=5,
                    help="Create a new dataset. Takes 5 arguments: "
                         "[Directory with classes: Other/Classes/Carreteras] "
                         "[Resize dimensions: 140x80] "
                         "[Crop dimensions Left,Top,Right,Bottom: 0:0:140:80] "
                         "[Percentage of frames used for training: 0.75] "
                         "[Percentage of frames used for testing: 0.15]")

parser.add_argument("-n", "--network", type=int, help="Create a CNN with this many iterations")
parser.add_argument("-p", "--play", help="Use this video to play the loaded CNN")
parser.add_argument("output", help="Directory where to save and load the generated data.")

args = parser.parse_args()

data_set = None

if args.dataset:
    classes_root = args.dataset[0]
    resize = tuple(map(int, args.dataset[1].split("x")))
    crop = tuple(map(int, args.dataset[2].split(":")))
    train = float(args.dataset[3])
    if not 0 <= train <= 1:
        print("Train percentage must be between 0 and 1.", file=stderr)
        exit(1)
    test = float(args.dataset[4])
    if not 0 <= test <= 1:
        print("Test percentage must be between 0 and 1.", file=stderr)
        exit(1)
    if not 0 <= test + train <= 1:
        print("Test + train percentages must be between 0 and 1.", file=stderr)
        exit(1)
    validation = 1 - train - test

    data_set = DataSet.from_directory(classes_root, resize, crop, train, test)
    data_set.to_file(args.output)

if args.network:
    if not data_set:
        data_set = DataSet.from_file(args.output)
    create_cnn(data_set, args.output, args.network)

if args.play:
    meta = data_set.get_metadata() if data_set else DataSet.get_meta(args.output)
    play_cnn(meta, args.output, args.play)
