import imageio


class Video:
    def __init__(self, filename):
        print("Leyendo " + filename)
        self.vid = imageio.get_reader(filename)
        self.len = len(self.vid)
        self.current = -1
        print("Leídos %d frames de %s." % (self.len, filename))

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current >= self.len:
            raise StopIteration
        else:
            return self.vid.get_data(self.current)

    def __len__(self):
        return self.len
