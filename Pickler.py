import pickle as p
import os
from os.path import join

pickle_protocol = 2
train_name = "train_objects"
test_name = "test_objects"
validation_name = "validation_objects"


def pickle(dataset, output_directory, train_pct=0.6, test_pct=0.2, validation_pct=0.2):
    items_for_training = int(train_pct * len(dataset))
    items_form_testing = int(test_pct * len(dataset))
    items_for_validation = int(validation_pct * len(dataset))

    p.dump(dataset.get_next(items_for_training), open(os.path.join(output_directory, train_name), "wb"),
           pickle_protocol)
    p.dump(dataset.get_next(items_form_testing), open(os.path.join(output_directory, test_name), "wb"),
           pickle_protocol)
    p.dump(dataset.get_next(items_for_validation), open(os.path.join(output_directory, validation_name), "wb"),
           pickle_protocol)


def unpickle(directory):
    if os.access(join(directory, train_name), os.F_OK) and os.access(join(directory, test_name), os.F_OK) and os.access(
            join(directory, validation_name), os.F_OK):

        return p.load(open(os.path.join(directory, train_name), "rb")), \
            p.load(open(os.path.join(directory, test_name), "rb")), \
            p.load(open(os.path.join(directory, validation_name), "rb"))
