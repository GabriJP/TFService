# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import imageio
from PIL import Image
import numpy as np
import scipy.misc as misc
import scipy.ndimage as ndim


class Video:

    def __init__(self, filename, size=(300, 300), crop=(0, 0, 300, 300)):
        print("Leyendo " + filename)
        self.vid = imageio.get_reader(filename)
        self.size = size
        self.crop = crop
        self.len = len(self.vid)
        self.current = -1
        print("Leídos %d frames de %s." % (self.len, filename))

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current >= self.len or self.current == 10:
            raise StopIteration
        else:
            img = Image.fromarray(self.vid.get_data(self.current))
            img = img.resize(self.size, Image.ANTIALIAS)
            img = img.crop(self.crop)
            return misc.fromimage(misc.toimage(img), flatten=True)

    next = __next__

    def __len__(self):
        return self.len
