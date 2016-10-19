# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
from GUI import *
import Pickler

n = getclassnum()
clases = []

for i in range(1, n + 1):
    clases.append((getclassname(i), getclassdirectory(i)))

redimensions = getredimension()
crop = getcrop()
percentages = getpercentages()
output = getoutputdrectory()

Pickler.pickle(clases, "output")
