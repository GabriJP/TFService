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
from os.path import join
from io import BytesIO

pickled_dataset_filename = "dataset"
pickle_protocol = 2


class DataSet:
    def __init__(self, train_set, test_set, validation_set, number_of_classes, train_pct=0.85, test_pct=0.15):
        if train_pct + test_pct > 1:
            raise AttributeError("train_pct + test_pct must be less than or equal to 1")
        self.train = train_set
        self.test = test_set
        self.validation = validation_set
        self.number_of_classes = number_of_classes
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.validation_pct = 1 - train_pct - test_pct
        self.training_index = 0
        self.test_index = 0
        self.validation_index = 0

    @classmethod
    def from_videos(cls, videos, new_dimensions, crop_dimensions, train_pct, test_pct):
        pool = ThreadPool()
        results = [pool.apply_async(cls.get_frames_from_video, (video, new_dimensions, crop_dimensions)) for
                   label, video in videos]
        pool.close()
        pool.join()
        frames = [(label_file[0], frame) for label_file, thread_async_result in zip(videos, results) for frame in
                  thread_async_result.get()]
        np.random.shuffle(frames)

        train_index = int(len(frames) * train_pct)
        test_index = int(len(frames) * test_pct) + train_index

        train_set = frames[:train_index]
        test_set = frames[train_index:test_index]
        validation_set = frames[test_index:]

        return cls(train_set, test_set, validation_set, len(set(list(zip(*videos))[0])), train_pct, test_pct)

    @staticmethod
    def get_frames_from_video(filename, new_dimensions, crop_dimensions):
        vid = imageio.get_reader(filename)
        first_frame = vid.get_data(0)
        if first_frame.shape[0] / new_dimensions[0] != first_frame.shape[1] / new_dimensions[1]:
            print("Warning: aspect ratio will be lost in resize process for %s" % filename, file=stderr)
        if crop_dimensions[2] - crop_dimensions[0] > new_dimensions[0] or crop_dimensions[3] - crop_dimensions[1] > \
                new_dimensions[1]:
            raise ValueError("Crop size bigger than resize")

        func = (lambda x: Image.fromarray(x).convert('L').resize(new_dimensions, Image.ANTIALIAS).crop(crop_dimensions))
        return list(map(func, vid))

    @classmethod
    def from_file(cls, directory):
        if not os.access(join(directory, pickled_dataset_filename), os.F_OK):
            raise IOError("Dataset file not found")

        file = gzip.open(join(directory, pickled_dataset_filename), "rb")
        dictionary = p.load(file)
        file.close()

        func = (lambda y: list(map((lambda x: (x[0], cls.image_base64_to_numpy(x[1]))), y)))

        train_set = func(dictionary['train_set'])
        test_set = func(dictionary['test_set'])
        validation_set = func(dictionary['validation_set'])

        return cls(train_set, test_set, validation_set, dictionary['n_classes'], dictionary['train_pct'],
                   dictionary['test_pct'])

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

        dictionary = {'train_set': train_set, 'test_set': test_set, 'validation_set': validation_set,
                      'n_classes': self.number_of_classes, 'train_pct': self.train_pct, 'test_pct': self.test_pct}

        file = gzip.open(join(directory, pickled_dataset_filename), "wb")
        p.dump(dictionary, file, pickle_protocol)
        file.close()

    @staticmethod
    def image_numpy_to_base64(image):
        cache = BytesIO()
        image.save(cache, format="JPEG")
        return base64.b64encode(cache.getvalue())

    def next_training_batch(self, size=10):
        if self.training_index + size >= len(self.train):
            self.training_index = 0
        self.training_index += size
        return tuple(zip(*self.train[self.training_index - size:self.training_index]))

    def next_test_batch(self, size=10):
        if self.test_index + size >= len(self.test):
            self.test_index = 0
        self.test_index += size
        return tuple(zip(*self.test[self.test_index - size:self.test_index]))

    def next_validation_batch(self, size=10):
        if self.validation_index + size >= len(self.validation):
            self.validation_index = 0
        self.validation_index += size
        return tuple(zip(*self.validation[self.validation_index - size:self.validation_index]))

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
        return len(self.get_classes())

    def get_classes(self):
        label_set = set()
        for label, frame in self.train:
            label_set.add(label)
        for label, frame in self.test:
            label_set.add(label)
        for label, frame in self.validation:
            label_set.add(label)
        return list(label_set)

    def frame_pixels(self):
        return self.train[0][1].size
