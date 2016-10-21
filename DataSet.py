import numpy as np
from Video import Video


class DataSet:
    def __init__(self, videos: [(str, Video)] = None):
        """
        Builds a DataSet

        :param videos: list of tuples (label, video)
        """
        if videos is None:
            videos = []
        self.current = 0
        self.frames = [(current_labeled_video[0], frame) for current_labeled_video in videos for frame in current_labeled_video[1]]
        np.random.shuffle(self.frames)

    def __iter__(self):
        return self.frames.__iter__()

    def __len__(self):
        return len(self.frames)

    def add_video(self, label: str, v: Video):
        """
        Adds labeled video frames to this object

        :param label: Label for the video
        :param v: video returning a ready-to-use frame on each iteration
        """
        self.frames.extend([(label, frame) for frame in v])
        np.random.shuffle(self.frames)

    def get_next(self, n: int = 10):
        """
        Returns a list of frames of size n

        :param n: Max size of the list to return
        :return: A list of tuples (label, frame) of size 0-n. If returned list is not size n, there are not more
        elements
        """
        self.current += n
        return self.frames[self.current - n:self.current]
