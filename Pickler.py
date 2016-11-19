# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import base64
import os
import gzip
import pickle as p
import numpy as np

from PIL import Image
from io import BytesIO
from os.path import join

from DataSet import DataSet

pickle_protocol = -1
frames_name = "frames_data"
control_data_name = "control_data"


def pickle(data_set, directory):
    if not os.path.exists(directory):
        os.makedirs(directory, 0o0755)

    control_data_file = gzip.open(join(directory, control_data_name), "wb")
    p.dump(data_set.get_control_data(), control_data_file, pickle_protocol)
    control_data_file.close()

    frames_file = gzip.open(join(directory, frames_name), "wb")
    p.dump([(label, to_base64(frame)) for label, frame in data_set.get_all_labelled_frames()], frames_file,
           pickle_protocol)
    frames_file.close()


def to_base64(image):
    cache = BytesIO()
    image.save(cache, format="JPEG")
    return base64.b64encode(cache.getvalue())


def unpickle(directory):
    if not os.access(join(directory, frames_name), os.F_OK):
        raise IOError("frames file not found")

    if not os.access(join(directory, control_data_name), os.F_OK):
        raise IOError("control data file not found")

    control_data_file = gzip.open(join(directory, control_data_name), "rb")
    control_data = p.load(control_data_file)
    control_data_file.close()

    frames_file = gzip.open(join(directory, frames_name), "rb")
    frames = p.load(frames_file)
    frames_file.close()

    return DataSet([(label, from_base64(frame)) for label, frame in frames], control_data[0], control_data[1],
                   control_data[2])


def from_base64(image):
    return np.asarray(Image.open(BytesIO(base64.b64decode(image))))
