from numpy.random import permutation as perm

from . import utils

class LoadMethod:
    """ Skeleton class for loading data from memory.
    As a minimum it must include following functionality:
    - load/read data from memory (mandatory)
    - pre-process the read data
    - fetch next set of samples
    """

    def __init__(self):
        """ Constructor where we need to pass all needed arguments such as
        relevant memory paths, wanted sequence lengths, etc.
        Furthermore, we implicitly load and pre-process data.
        """
        self._load_data()
        self._preprocess_data()


    def __call__(self, idx):
        """ Fetch specific tuple of input and target. """
        return self.samples[idx]


    def _load_data(self):
        """ Load data from memory.
        List of samples should an instance variable of a zip object, e.g.: 
        
          train_X, train_t = [], []
          self.samples = list(zip(train_X, train_t))
        """
        raise NotImplementedError("Please implement `_load_data()`")


    def _preprocess_data(self):
        """ Pre-process read data, e.g. strip whitespace, filter samples by
        length, sort samples, augment samples, etc.
        """
        print("Data pre-processing functionality has not been implemented..")
        print("  No pre-processing will be performed..")
        pass


class SampleGenerator:
    """ Purpose is to fetch a sample from a LoadMethod instance and yield the 
    result.
    """

    def __init__(self, load_method, permutation=None, shuffle=False,
            repeat=False):
        """ Mandatory LoadMethod instance.
        Optional arguments:
        permutation -- a list of indices to permute over
        shuffle -- whether or not to shuffle list to iterate over
        repeat -- whether or not to repeat yielding samples from LoadMethod
        """
        self.permutation = permutation or range(len(load_method.samples))
        self.load_method = load_method
        self.shuffle = shuffle
        self.repeat = repeat


    def gen_sample(self):
        """ Fetch samples from LoadMethod by iterating over indices of
        permutation list.
        If we are training, shuffling should be enabled.
        List of permutation can be shuffled.
        """
        while True:
            if self.shuffle:
                self.permutation = perm(self.permutation)
            for i in range(len(self.permutation)):
                yield self.load_method(self.permutation[i])
            if not self.repeat:
                break


class BatchGenerator:

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

