# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
# from VideoSplit import videotoframes
# import os
#
# clases = []
#
# # nClases = input('Número de clases: ')
# nClases = 1
#
# for i in range(1, nClases + 1, 1):
#     # clases.append(raw_input('Directorio de la clase ' + str(i) + ': '))
#     clases.append('/home/gabriel/Escritorio/1')
#
# # tmpDir = raw_input('Inserte directorio temporal: ')
# tmpDir = '/home/gabriel/Escritorio/tmp'
#
# count = 1
#
# for clasPath in clases:
#     clasName = clasPath[clasPath.rfind('/')+1:]
#     outputDir = tmpDir + '/' + clasName
#     files = os.listdir(clasPath)
#
#     if not os.path.exists(outputDir):
#         os.makedirs(outputDir)
#
#     for currentFile in files:
#         count = videotoframes(str(clasPath) + "/" + str(currentFile), outputDir, "jpg", count)


from Video import Video
import matplotlib.pyplot as plt

size = 300, 300

vid = Video("big_buck_bunny_720p_5mb.mp4")
for img in vid:
    plt.imshow(img, interpolation='nearest')
    plt.show()
    break
