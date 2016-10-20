# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
from GUI import *
from video import video
from dataset import dataset

n = getclassnum()
clases = []

for i in range(1, n + 1):
    clases.append((getclassname(i), getclassdirectory(i)))

redimensions = getredimension()
crop = getcrop()
percentages = getpercentages()
output = getoutputdrectory()

dataSet = dataset([(class_name, video(path, redimensions, crop)) for class_name, path in clases])

