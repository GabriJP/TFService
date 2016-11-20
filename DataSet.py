# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import base64
import os
import gzip

import numpy as np
import pickle as p

from multiprocessing.pool import ThreadPool
from os.path import join
from PIL import Image
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
    def from_videos(cls, videos, train_pct, test_pct):
        pool = ThreadPool()
        results = [pool.apply_async(video.get_frames) for label, video in videos]
        pool.close()
        pool.join()
        frames = [(label[0], frame) for label, thread_async_result in zip(videos, results) for frame in
                  thread_async_result.get()]
        np.random.shuffle(frames)

        train_index = int(len(frames) * train_pct)
        test_index = int(len(frames) * test_pct) + train_index

        train_set = frames[:train_index]
        test_set = frames[train_index:test_index]
        validation_set = frames[test_index:]

        return cls(train_set, test_set, validation_set, len(set(list(zip(*videos))[0])), train_pct, test_pct)

    @classmethod
    def from_file(cls, directory):
        if not os.access(join(directory, pickled_dataset_filename), os.F_OK):
            raise IOError("Dataset file not found")

        file = gzip.open(join(directory, pickled_dataset_filename), "rb")
        dictionary = p.load(file)
        file.close()

        train_set = cls.list_base64_to_numpy(dictionary['train_set'])
        test_set = cls.list_base64_to_numpy(dictionary['test_set'])
        validation_set = cls.list_base64_to_numpy(dictionary['validation_set'])

        return cls(train_set, test_set, validation_set, dictionary['n_classes'], dictionary['train_pct'],
                   dictionary['test_pct'])

    @classmethod
    def list_base64_to_numpy(cls, labelled_frames):
        return [(label, cls.image_base64_to_numpy(frame)) for label, frame in labelled_frames]

    @staticmethod
    def image_base64_to_numpy(image):
        return np.asarray(Image.open(BytesIO(base64.b64decode(image))))

    def to_file(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory, 0o0755)

        train_set = self.list_numpy_to_base64(self.train)
        test_set = self.list_numpy_to_base64(self.test)
        validation_set = self.list_numpy_to_base64(self.validation)

        dictionary = {'train_set': train_set, 'test_set': test_set, 'validation_set': validation_set,
                      'n_classes': self.number_of_classes, 'train_pct': self.train_pct, 'test_pct': self.test_pct}

        file = gzip.open(join(directory, pickled_dataset_filename), "wb")
        p.dump(dictionary, file, pickle_protocol)
        file.close()

    @classmethod
    def list_numpy_to_base64(cls, labelled_frames):
        return [(label, cls.image_numpy_to_base64(frame)) for label, frame in labelled_frames]

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
        result = {}
        labels = list(zip(*self.train))[0]
        labels.extend(list(zip(*self.test))[0])
        labels.extend(list(zip(*self.validation))[0])
        for label in labels:
            if label not in result:
                result[label] = 1
            else:
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
        return label_set

    def frame_pixels(self):
        return self.train[0][1].size
