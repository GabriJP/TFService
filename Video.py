# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import imageio

from PIL import Image
from sys import stderr


class Video:
    def __init__(self, filename, new_dimensions, crop_dimensions):
        self.filename = filename
        self.vid = imageio.get_reader(filename)
        self.new_dimensions = new_dimensions
        self.crop_dimensions = crop_dimensions
        self.len = len(self.vid)
        self.current = -1
        first_frame = self.vid.get_data(0)
        if first_frame.shape[0] / new_dimensions[0] != first_frame.shape[1] / new_dimensions[1]:
            print("Warning: aspect ratio will be lost in resize process for %s" % filename, file=stderr)
        if crop_dimensions[2] - crop_dimensions[0] > new_dimensions[0] or crop_dimensions[3] - crop_dimensions[1] > \
                new_dimensions[1]:
            raise ValueError("Crop size bigger than resize")

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current >= self.len:
            raise StopIteration
        else:
            try:
                img = Image.fromarray(self.vid.get_data(self.current)).convert('L')
                img = img.resize(self.new_dimensions, Image.ANTIALIAS)
                img = img.crop(self.crop_dimensions)
                return img
            except RuntimeError:
                print('Error reading "%s", ignoring error...' % self.filename, file=stderr)
                raise StopIteration

    next = __next__

    def __len__(self):
        return self.len

    def get_frames(self):
        """

        :return: List of frames of this video
        :rtype: list(ndarray)
        """
        return [frame for frame in self]
