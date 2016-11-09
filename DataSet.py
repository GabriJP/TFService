# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from multiprocessing.pool import ThreadPool


class DataSet:
    def __init__(self, videos=None, new_frames=None):
        """

        :param videos: list of tuples (label, video)
        """
        if videos is None:
            self.frames = []
        else:
            pool = ThreadPool()
            results = [pool.apply_async(video.get_frames) for label, video in videos]
            pool.close()
            pool.join()
            self.frames = [(label[0], frame) for label, thread_async_result in zip(videos, results) for frame in
                           thread_async_result.get()]
            np.random.shuffle(self.frames)
        if new_frames is not None:
            self.frames.extend(new_frames)
        self.current = 0

    def __iter__(self):
        return self.frames.__iter__()

    def __len__(self):
        return len(self.frames)

    def add_video(self, label, v):
        """
        Adds a video to this set so its frames can be returned with the provided label

        :param label: Label for the video
        :param v: video returning a ready-to-use frame on each iteration
        """
        self.frames.extend([(label, frame) for frame in v])
        np.random.shuffle(self.frames)

    def next_batch(self, n=10):
        self.current += n
        return tuple(zip(*self.frames[self.current - n:self.current]))

    def get_frames_per_label(self):
        result = {}
        for label, frame in self.frames:
            if label not in result:
                result[label] = 1
            else:
                result[label] += 1

        return result

    def add_all(self, frame_list):
        self.frames.extend(frame_list)
