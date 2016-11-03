import pickle as p
import os
from DataSet import DataSet
from os.path import join

pickle_protocol = 2
train_name = "train_objects"
test_name = "test_objects"
validation_name = "validation_objects"


def pickle(data_set, output_directory, train_pct=0.6, test_pct=0.2, validation_pct=0.2):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, 0o0755)
    items_for_training = int(train_pct * len(data_set))
    items_form_testing = int(test_pct * len(data_set))
    items_for_validation = int(validation_pct * len(data_set))

    p.dump(data_set.get_next(items_for_training), open(os.path.join(output_directory, train_name), "wb"),
           pickle_protocol)
    p.dump(data_set.get_next(items_form_testing), open(os.path.join(output_directory, test_name), "wb"),
           pickle_protocol)
    p.dump(data_set.get_next(items_for_validation), open(os.path.join(output_directory, validation_name), "wb"),
           pickle_protocol)


def unpickle(directory):
    if os.access(join(directory, train_name), os.F_OK) and os.access(join(directory, test_name), os.F_OK) and os.access(
            join(directory, validation_name), os.F_OK):
        return DataSet(new_frames=p.load(open(os.path.join(directory, train_name), "rb"))), DataSet(
            new_frames=p.load(open(os.path.join(directory, test_name), "rb"))), DataSet(
            new_frames=p.load(open(os.path.join(directory, validation_name), "rb")))
