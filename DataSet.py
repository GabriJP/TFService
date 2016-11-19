# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from multiprocessing.pool import ThreadPool


class DataSet:
    def __init__(self, labelled_frames, number_of_classes, train_pct=0.85, test_pct=0.15):
        if train_pct + test_pct > 1:
            raise AttributeError("train_pct + test_pct must be less than or equal to 1")
        self.number_of_classes = number_of_classes
        self.train_pct = train_pct
        self.test_pct = test_pct
        self.validation_pct = 1 - train_pct - test_pct
        self.training_index = 0
        self.test_index = 0
        self.validation_index = 0
        train_index = int(len(labelled_frames) * train_pct)
        test_index = int(len(labelled_frames) * test_pct) + train_index
        self.train = labelled_frames[:train_index]
        self.test = labelled_frames[train_index:test_index]
        self.validation = labelled_frames[test_index:]

    @classmethod
    def from_videos(cls, videos, train_pct, test_pct):
        pool = ThreadPool()
        results = [pool.apply_async(video.get_frames) for label, video in videos]
        pool.close()
        pool.join()
        frames = [(label[0], frame) for label, thread_async_result in zip(videos, results) for frame in
                  thread_async_result.get()]
        np.random.shuffle(frames)

        return cls(frames, len(set(list(zip(*videos))[0])), train_pct, test_pct)

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

    def add_video(self, label, v):
        self.add_all([(label, frame) for frame in v])

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

    def add_all(self, labelled_frames):
        train_index = int(len(labelled_frames) * self.train_pct)
        test_index = int(len(labelled_frames) * self.test_pct) + train_index
        self.train = labelled_frames[:train_index]
        self.test = labelled_frames[train_index:test_index]
        self.validation = labelled_frames[test_index:]
        self.train.extend(labelled_frames[:len(labelled_frames) * self.train_pct])
        self.test.extend(labelled_frames[len(labelled_frames) * self.train_pct:len(labelled_frames) * self.test_pct])
        self.validation.extend(labelled_frames[len(labelled_frames) * self.test_pct:])
        np.random.shuffle(self.train)
        np.random.shuffle(self.test)
        np.random.shuffle(self.validation)

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

    def frame_size(self):
        return self.train[0][1].shape

    def frame_pixels(self):
        return self.train[0][1].size

    def get_all_labelled_frames(self):
        result = self.train
        result.extend(self.test)
        result.extend(self.validation)
        return result

    def get_control_data(self):
        return self.number_of_classes, self.train_pct, self.test_pct
