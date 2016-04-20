from numpy.random import permutation as permute

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


class SampleGenerator:
    """ Fetch a sample from a Loader instance and yield the result. """

    def __init__(self, loader, indexes=None, shuffle=False, repeat=False):
        """ Mandatory Loader instance.
        Optional arguments:
        indexes -- a list of indexes to permute over
        shuffle -- whether or not to shuffle list to iterate over
        repeat -- whether or not to repeat yielding samples from Loader
        """
        self.indexes = indexes if indexes is not None else range(
                len(loader.samples))
        self.loader = loader
        self.shuffle = shuffle
        self.repeat = repeat


    def gen_sample(self):
        """ Fetch samples from Loader by iterating over indexes list.
        If we are training, shuffling should be enabled.
        List of indexes can be shuffled.
        """
        while True:
            if self.shuffle:
                self.indexes = permute(self.indexes)
            for i in range(len(self.indexes)):
                yield self.loader(self.indexes[i])
            if not self.repeat:
                break


class BatchGenerator:
    """ Purpose is to generate batches of samples from a SampleGenerator. """

    def __init__(self, sample_generator, batch_size):
        """ Must have a SampleGenerator and wanted batch size. """
        self.sample_generator = sample_generator
        self.batch_size = batch_size
        self.samples = [] # make sure it is defined


    def _make_batch(self):
        """ Format samples as needed. """
        raise NotImplementedError("Please implement `_make_batch()`")


    def gen_batch(self):
        """ Generate a batch of max `batch_size` length. 
        Initialises samples to empty list. 
        Calls SampleGenerator to fetch samples from Loader.
        Yields SampleGenerator's `_make_batch` to format batch. 
        """
        self.samples = [] # make sure to reset list
        for sample in self.sample_generator.gen_sample():
            self.samples.append(sample)
            if len(self.samples) == self.batch_size:
                yield self._make_batch()
                self.samples = []  # reset batch
        # if samples is not empty, we want to return it.
        # NOTE don't reset samples, as we check size of latest batch
        if len(self.samples) > 0:
            yield self._make_batch()


    @property
    def latest_batch_size(self):
        """Actual size of the most recently produced batch.
        This is useful if there wasn't enough samples left in the data set to 
        make a full size batch.
        """
        return len(self.samples)
