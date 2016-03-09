import numpy as np
import os
import utils

from . import utils


# Method YOU should implement!
class LoadMethod(object):

    def __init__(self):
        self._prepare_data()

    def _load_data(self):
        self.train_X = []
        self.train_t = []
        self.samples = zip(self.train_X, self.train_t)

    def _preprocess_data(self):
        pass

    def _prepare_data(self):
        if not os.path.exists("data/train.npy.gz"):
            self._load_data()
            self._preprocess_data()
        else:
            self.samples = utils.load_gz("data/train.npy.gz")

    def __call__(self, idx):
        return self.samples[idx]


class SampleGenerator(object):

    def __init__(self, load_method, shuffle=False, repeat=False):
        self.load_method = load_method
        self.num_samples = len(self.load_method.samples)
        self.shuffle = shuffle
        self.repeat = repeat

        # This default permutation is only used if shuffle == False
        self.permutation = range(self.num_samples)

        print("ElemGenerator initiated")

    def gen_sample(self):
        while True:
            if self.shuffle:
                num_samples = self.num_samples
                self.permutation = np.random.permutation(num_samples)
            for num in range(self.num_samples):
                yield self.load_method(self.permutation[num])
            if not self.repeat:
                break


class BatchGenerator(object):

    def __init__(self, sample_generator, batch_size):
        self.sample_generator = sample_generator
        self.batch_size = batch_size
        self.samples = []

    def _make_batch(self):
        raise NotImplementedError

    def gen_batch(self):
        self.samples = []
        for sample in self.sample_generator.gen_sample():
            self.samples.append(sample)
            if len(self.samples) == self.batch_size:
                yield self._make_batch()
                self.samples = []  # reset batch

        # make a smaller batch from any remaining samples
        if len(self.samples) > 0:
            yield self._make_batch()

    @property
    def latest_batch_size(self):
        """Actual size of the most recently produced batch.

        This is useful if there weren't enough samples left in the data
        set to make a full size batch.
        """
        return len(self.samples)


class ChunkInfo(object):

    def __init__(self, chunk_size=4096, num_chunks=800):
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        print("ChunkInfo initiated")


class ChunkGenerator(object):

    def __init__(self, batch_generator, chunk_info, rng=np.random):
        self.batch_generator = batch_generator
        self.chunk_info = chunk_info
        self.rng = rng
        self.batches = []
        print("ChunkGenerator initiated")

    def _make_chunk_holder(self):
        self.chunk = []  # not in __init__ to reset it at every call
        self.chunk.append([])  # where the batch is located
        self.chunk.append(self.batch_idx)  # pasting the idx
        self.batch_idx = 0

    def _make_chunk(self):
        self._make_chunk_holder()
        for _ in range(len(self.batches)):  # smarter solution?
            self.chunk[0].append(self.batches.pop())
        return self.chunk

    def gen_chunk(self):
        self.batch_idx = 0
        for batch in self.batch_generator.gen_batch():
            self.batches.append(batch)
            self.batch_idx += 1
            if self.batch_idx >= self.chunk_info.chunk_size:
                yield self._make_chunk()
        if self.batch_idx > 0:
            yield self._make_chunk()
