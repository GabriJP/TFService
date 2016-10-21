import os
from Video import Video
import numpy as np
import pickle as p


def pickle(classes, output, train_percentage=0.8):
    """

    :param output: Output directory
    :param train_percentage: Percentage of data used for training
    :type classes: List of tuples (ClassName, ClassPath)
    """

    frames = [frame for classname, classpath in classes for frame in getclassframes(classname, classpath)]
    np.random.shuffle(frames)
    items_for_training = int(len(frames) * train_percentage)

    data_matrix_train = np.array([np.array(t[1]) for t in frames[:items_for_training]])
    label_list_train = np.array([t[0] for t in frames[:items_for_training]])
    data_matrix_test = np.array([np.array(t[1]) for t in frames[items_for_training:]])
    label_list_test = np.array([t[0] for t in frames[items_for_training:]])

    train = {"data": data_matrix_train.astype("int"), "labels": label_list_train.astype("int")}
    test = {"data": data_matrix_test.astype("int"), "labels": label_list_test.astype("int")}

    p.dump(train, open(os.path.join(output, "data_objects"), "wb"))
    p.dump(test, open(os.path.join(output, "test_objects"), "wb"))


def getvideotaggedframes(classname, video):
    """

    :type classname: Name of the class of the video
    :type video: Video
    """
    return [(classname, frame) for frame in video]


def getfilesinpath(path):
    return [os.path.join(dirpath, filename) for dirpath, dirnames, filenames in os.walk(path) for filename in filenames]


def getclassframes(classname, classpath):
    return [frame for file in getfilesinpath(classpath) for frame in getvideotaggedframes(classname, Video(file))]
