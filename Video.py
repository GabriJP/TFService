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

    def get_frames(self):
        """

        :return: List of frames of this video
        :rtype: list(ndarray)
        """
        result = []
        for frame in self.vid:
            img = Image.fromarray(frame).convert('L')
            img = img.resize(self.new_dimensions, Image.ANTIALIAS)
            img = img.crop(self.crop_dimensions)
            result.append(img)
        return result
