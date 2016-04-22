import math
import numpy as np


class Loader:
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


class IterationSchedule:
    def __init__(self, shuffle=False, repeat=False):
        self.shuffle = shuffle
        self.repeat = repeat

    def gen_indices(self, loader, batch_size):
        """Yields lists of indices that make up batches.

        Make batches using the lists of indices that this function yields
        by picking the samples from the loader that have the given indices.

        Keyword arguments:
        loader -- Loader whose samples should be iterated over.
        batch_size -- Desired batch size.
        """
        num_samples = len(loader.samples)
        num_batches = math.ceil(num_samples/batch_size)

        batch_numbers = range(num_batches)
        while True:
            if self.shuffle:
                batch_numbers = np.random.permutation(num_batches)
            for batch in batch_numbers:
                yield range(batch*batch_size, min((batch+1)*batch_size, num_samples-1))

            if not self.repeat:
                break


class BatchGenerator:
    """ Purpose is to generate batches of samples from a Loader. """

    def __init__(self, loader, batch_size, iteration_schedule=None):
        """Must have a loader, wanted batch size, and an iteration schedule."""
        self.loader = loader
        self.iteration_schedule = iteration_schedule or IterationSchedule()
        self.batch_size = batch_size
        self.samples = []  # just to make sure variable is defined

    def _make_batch(self):
        """ Format samples as needed. """
        raise NotImplementedError("Please implement `_make_batch()`")

    def gen_batch(self):
        """ Generate a batch of max `batch_size` length.
        Initialises samples to empty list.
        Fetches samples from Loader w.r.t. indices from iteration schedule.
        Yields formatted batches from BatchGenerator's `_make_batch`.
        """
        for indices in self.iteration_schedule.gen_indices(self.loader, self.batch_size):
            self.samples = []  # make sure to reset list
            for index in indices:
                self.samples.append(self.loader.samples[index])
            yield self._make_batch()

    @property
    def latest_batch_size(self):
        """Actual size of the most recently produced batch.
        This is useful if there wasn't enough samples left in the data set to
        make a full size batch.
        """
        return len(self.samples)
