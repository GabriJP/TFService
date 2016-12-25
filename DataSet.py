# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import base64
import gzip
from collections import defaultdict

import imageio
import os

import numpy as np
import pickle as p

from multiprocessing.pool import ThreadPool
from PIL import Image
from sys import stderr
from io import BytesIO
from os.path import join, isfile
from os import listdir

pickled_dataset_filename = "dataset.dump"
pickled_meta_dataset_filename = "meta_dataset.dump"
pickle_protocol = 2


class DataSet:
    def __init__(self, train_set, test_set, validation_set, number_of_classes, shape, labels, train_pct=0.85,
                 test_pct=0.15):

        if train_pct + test_pct > 1:
            raise AttributeError("train_pct + test_pct must be less than or equal to 1")
        self.train = train_set
        self.test = test_set
        self.validation = validation_set
        self.number_of_classes = number_of_classes
        self.shape = shape
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.validation_pct = 1 - train_pct - test_pct
        self.training_index = 0
        self.test_index = 0
        self.validation_index = 0
        self.labels = labels
        labels = self.one_hot(list(range(number_of_classes)), number_of_classes)
        self.labels = dict(zip(self.get_classes(), labels))

    @classmethod
    def from_directory(cls, root_directory, new_dimensions, crop_dimensions, train_pct, test_pct):
        directories = {directory: join(root_directory, directory) for directory in listdir(root_directory) if
                       not isfile(join(root_directory, directory))}

        videos = defaultdict(dict)

        for label, directory in directories.items():
            files = [join(directory, file) for file in listdir(directory) if isfile(join(directory, file))]
            videos[label] = files

        return cls.from_videos(videos.items(), new_dimensions, crop_dimensions, train_pct, test_pct)

    @classmethod
    def from_videos(cls, label_videos, new_dimensions, crop_dimensions, train_pct, test_pct):
        pool = ThreadPool()
        results = [pool.apply_async(cls.get_frames_from_video, (video, new_dimensions, crop_dimensions)) for
                   label, videos in label_videos for video in videos]
        pool.close()
        pool.join()
        frames = [(label_file[0], frame) for label_file, thread_async_result in zip(label_videos, results) for frame in
                  thread_async_result.get()]
        np.random.shuffle(frames)

        train_index = int(len(frames) * train_pct)
        test_index = int(len(frames) * test_pct) + train_index

        train_set = frames[:train_index]
        test_set = frames[train_index:test_index]
        validation_set = frames[test_index:]

        number_of_classes = len(set(list(zip(*label_videos))[0]))

        label_set = set()
        for label, frame in train_set:
            label_set.add(label)
        for label, frame in test_set:
            label_set.add(label)
        for label, frame in validation_set:
            label_set.add(label)

        network_expected_outputs = cls.one_hot(list(range(number_of_classes)), number_of_classes)
        label_with_expected_output = dict(zip(label_set, network_expected_outputs))

        func = (lambda labelled_frame: (label_with_expected_output.get(labelled_frame[0]), labelled_frame[1]))
        train_set = list(map(func, train_set))
        test_set = list(map(func, test_set))
        validation_set = list(map(func, validation_set))

        return cls(train_set, test_set, validation_set, number_of_classes, frames[0][1].shape,
                   label_with_expected_output, train_pct, test_pct)

    @classmethod
    def get_frames_from_video(cls, filename, new_dimensions, crop_dimensions):
        vid = imageio.get_reader(filename)
        first_frame = vid.get_data(0)
        if first_frame.shape[0] / new_dimensions[0] != first_frame.shape[1] / new_dimensions[1]:
            print("Warning: aspect ratio will be lost in resize process for %s" % filename, file=stderr)
        if crop_dimensions[2] - crop_dimensions[0] > new_dimensions[0] or crop_dimensions[3] - crop_dimensions[1] > \
                new_dimensions[1]:
            raise ValueError("Crop size bigger than resize")

        func = (lambda x: cls.process_frame(x, new_dimensions, crop_dimensions))
        return list(map(func, vid))

    @staticmethod
    def process_frame(frame, new_dimensions, crop_dimensions):
        return np.asarray(
            Image.fromarray(frame).convert('L').resize(new_dimensions, Image.ANTIALIAS).crop(crop_dimensions))

    @classmethod
    def from_file(cls, directory):
        if not os.access(join(directory, pickled_dataset_filename), os.F_OK):
            raise IOError("Dataset file not found")

        file = gzip.open(join(directory, pickled_dataset_filename), "rb")
        dictionary = p.load(file)
        file.close()

        file = gzip.open(join(directory, pickled_meta_dataset_filename), "rb")
        meta_dictionary = p.load(file)
        file.close()

        func = (lambda y: list(map((lambda x: (x[0], cls.image_base64_to_numpy(x[1]))), y)))

        train_set = func(dictionary['train_set'])
        test_set = func(dictionary['test_set'])
        validation_set = func(dictionary['validation_set'])

        return cls(train_set,
                   test_set,
                   validation_set,
                   meta_dictionary['n_classes'],
                   meta_dictionary['shape'],
                   meta_dictionary['labels'],
                   meta_dictionary['train_pct'],
                   meta_dictionary['test_pct'])

    @staticmethod
    def image_base64_to_numpy(image):
        return np.asarray(Image.open(BytesIO(base64.b64decode(image))))

    def to_file(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory, 0o0755)

        func = (lambda y: list(map((lambda x: (x[0], self.image_numpy_to_base64(x[1]))), y)))

        train_set = func(self.train)
        test_set = func(self.test)
        validation_set = func(self.validation)

        dictionary = {'train_set': train_set,
                      'test_set': test_set,
                      'validation_set': validation_set,
                      'n_classes': self.number_of_classes,
                      'train_pct': self.train_pct,
                      'test_pct': self.test_pct}

        meta_dictionary = {'n_classes': self.number_of_classes,
                           'shape': self.shape,
                           'labels': self.labels,
                           'train_pct': self.train_pct,
                           'test_pct': self.test_pct,
                           'frame_pixels': self.frame_pixels()}

        file = gzip.open(join(directory, pickled_dataset_filename), "wb")
        p.dump(dictionary, file, pickle_protocol)
        file.close()

        file = gzip.open(join(directory, pickled_meta_dataset_filename), "wb")
        p.dump(meta_dictionary, file, pickle_protocol)
        file.close()

    @staticmethod
    def image_numpy_to_base64(image):
        cache = BytesIO()
        image.save(cache, format="JPEG")
        return base64.b64encode(cache.getvalue())

    def next_training_batch(self, size=10):
        self.training_index, result = self.next_batch(self.train, self.training_index, size)
        return result

    def next_test_batch(self, size=10):
        self.test_index, result = self.next_batch(self.test, self.test_index, size)
        return result

    def next_validation_batch(self, size=10):
        self.validation_index, result = self.next_batch(self.validation, self.validation_index, size)
        return result

    @staticmethod
    def next_batch(dataset, index, size=10):
        if index + size >= len(dataset):
            index = 0
        index += size
        return index, tuple(zip(*dataset[index - size:index]))

    def get_frames_per_label(self):
        result = defaultdict(int)
        for label, frame in self.train:
            result[label] += 1
        for label, frame in self.test:
            result[label] += 1
        for label, frame in self.validation:
            result[label] += 1

        return result

    def get_number_of_classes(self):
        return self.number_of_classes

    def get_classes(self):
        return self.labels.keys()

    def frame_pixels(self):
        return self.shape[0] * self.shape[1]

    def get_frame_dimensions(self):
        return self.shape

    @staticmethod
    def get_meta(directory):
        if not os.access(join(directory, pickled_dataset_filename), os.F_OK) \
                or not os.access(join(directory, pickled_meta_dataset_filename), os.F_OK):
            raise IOError("Dataset file not found")

        file = gzip.open(join(directory, pickled_meta_dataset_filename), "rb")
        meta_dictionary = p.load(file)
        file.close()

        return meta_dictionary

    @staticmethod
    def one_hot(label_list, n):
        label_array = np.array(label_list).flatten()
        o_h = np.zeros((len(label_array), n))
        o_h[np.arange(len(label_array)), label_array - 1] = 1
        return o_h
