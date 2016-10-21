# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
import imageio
from PIL import Image
import scipy.misc as misc


class Video:
    def __init__(self, filename, size=(300, 300), crop=(0, 0, 300, 300)):
        self.vid = imageio.get_reader(filename)
        self.size = size
        self.crop = crop
        self.len = len(self.vid)
        self.current = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current >= self.len:
            raise StopIteration
        else:
            img = Image.fromarray(self.vid.get_data(self.current))
            img = img.resize(self.size, Image.ANTIALIAS)
            img = img.crop(self.crop)
            return misc.fromimage(misc.toimage(img), flatten=True)

    next = __next__

    def __len__(self):
        return self.len

    def getframes(self):
        return [frame for frame in self]
