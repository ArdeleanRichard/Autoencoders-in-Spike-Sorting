import numpy as np


class DataIterator:
    def __init__(self, data, BATCH_SIZE):
        self.data = np.array(data)
        self.BATCH_SIZE = BATCH_SIZE

    def __iter__(self):
        return self

    def __next__(self):
        random_indexes = np.random.randint(0,len(self.data), self.BATCH_SIZE)
        return self.data[random_indexes]