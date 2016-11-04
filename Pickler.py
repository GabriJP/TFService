import base64
import pickle as p
import os

import numpy as np
from PIL import Image
from io import BytesIO

from DataSet import DataSet
from os.path import join

pickle_protocol = -1
train_name = "train_objects"
test_name = "test_objects"
validation_name = "validation_objects"


def pickle(data_set, output_directory, train_pct=0.6, test_pct=0.2, validation_pct=0.2):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, 0o0755)

    items_for_training = int(train_pct * len(data_set))
    items_form_testing = int(test_pct * len(data_set))
    items_for_validation = int(validation_pct * len(data_set))

    save_to_file(data_set.next_batch(items_for_training), join(output_directory, train_name))
    save_to_file(data_set.next_batch(items_form_testing), join(output_directory, test_name))
    save_to_file(data_set.next_batch(items_for_validation), join(output_directory, validation_name))


def save_to_file(labelled_frames, path):
    p.dump([(label, to_base64(frame)) for label, frame in zip(*labelled_frames)], open(path, "wb"), pickle_protocol)


def to_base64(image):
    cache = BytesIO()
    image.save(cache, format="JPEG")
    return base64.b64encode(cache.getvalue())


def unpickle(directory):
    return load_from_file(join(directory, train_name)), load_from_file(join(directory, test_name)), load_from_file(
        join(directory, validation_name))


def load_from_file(path):
    if os.access(path, os.F_OK) and os.path.getsize(path) > 20:
        return DataSet(new_frames=[(label, from_base64(frame)) for label, frame in p.load(open(path, "rb"))])
    else:
        return DataSet()


def from_base64(image):
    return np.asarray(Image.open(BytesIO(base64.b64decode(image))))
