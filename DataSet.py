# coding=utf-8
import numpy as np
from multiprocessing.pool import ThreadPool


class DataSet:
    def __init__(self, videos=None):
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

    def get_next(self, n=10):
        """
        Returns a list of max size n with random frames

        :param n: Max size of the list to return
        :return: A list of tuples (label, frame) of size 0-n. If returned list is not size n, there are not more
        :rtype: list[(string, numpy.ndarray)]
        elements
        """
        self.current += n
        return self.frames[self.current - n:self.current]

    def get_frames_per_label(self):
        result = {}
        for label, frame in self.frames:
            if label not in result:
                result[label] = 1
            else:
                result[label] += 1

        return result