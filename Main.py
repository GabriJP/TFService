# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
from Video import Video
from DataSet import DataSet
from os import listdir
from os.path import isfile, join
from sys import argv

"""
Program usage:
python ProgramName.py class=class_name:class_directory [class=class_name:class_directory ...] \
ndim='new_height'x'new_width' crop=x_from:y_from:x_to:y_to train=percentage_for_training val=percentage_for_validation \
test=percentage_for_validation out=output_file

Example:
python ProgramName.py class=Mouse:Other/Classes/Mouse class=Pen:Other/Classes/Pen \
class=Scissors:Other/Classes/Scissors ndim=540x960 crop=20:230:520:730 \
train=0.75 val=0.15 test=0.1 out=Other/Output/Red.tf
"""

classes = []
for class_ in [arg[6:] for arg in argv if arg.startswith("class=")]:
    classes.append(tuple(class_.split(":")))

for current_ndim in [arg[5:] for arg in argv if arg.startswith("ndim=")]:
    new_dimensions = tuple([int(i) for i in current_ndim.split('x')])
    break

for current_crop in [arg[5:] for arg in argv if arg.startswith("crop=")]:
    crop_dimensions = tuple([int(i) for i in current_crop.split(':')])
    break

percentages = []
for current_train in [arg[6:] for arg in argv if arg.startswith("train=")]:
    percentages.append(current_train)
    break

for current_val in [arg[4:] for arg in argv if arg.startswith("val=")]:
    percentages.append(current_val)
    break

for current_test in [arg[5:] for arg in argv if arg.startswith("test=")]:
    percentages.append(current_test)
    break

percentages = tuple(percentages)

for output_current in [arg[5:] for arg in argv if arg.startswith("out=")]:
    output = output_current
    break

videos = []
for class_name, class_path in classes:
    for filename in [f for f in listdir(class_path) if isfile(join(class_path, f))]:
        videos.append((class_name, Video(join(class_path, filename), new_dimensions, crop_dimensions)))

data_set = DataSet(videos)
for label, frame in data_set:
    print(label)

frames_per_label = data_set.get_frames_per_label()
for label in frames_per_label:
    print('Frames for label "%s": %d' % (label, frames_per_label[label]))
