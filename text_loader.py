import math
import numpy as np

import frostings.loader as frost
from data.alphabet import Alphabet

def _filter_samples(samples, max_length):
    """Filter out samples of extreme length."""
    # remove input sentences that are too short or too long
    samples = [(x, t) for x, t in samples
               if len(x) > 1 and len(x) <= max_length-1]
    # remove target sentences that are too short or too long
    samples = [(x, t) for x, t in samples
               if len(t) > 1 and len(t) <= max_length-1]
    return list(set(samples))

def _truncate_samples(samples, limit):
    """Truncate long sentences."""
    return [(x[:limit], t[:limit]) for x, t in samples]

def _make_len_vec(sequences, offset=0, max_len=100000):
    """Get length of each sequence in list of sequences.

    Return as numpy vector.

    Keyword arguments:
    sequences -- list of sequences.
    offset -- (optional) value to be added to each length.
        Useful for including EOS characters.
    max_len -- (optional) (default:100,000) returned sequence length will
        not exceed this value.
    """
    return np.array([min(len(seq)+offset, max_len) for seq in sequences])


class TextLoader(frost.Loader):
    """Load and prepare text data."""

    def __init__(self, paths_X, paths_t, seq_len):
        """ Initialize TextLoader instance.

        Keyword arguments:
        paths_X -- list of paths for input sentences
        paths_t -- list of paths for target sentences
        seq_len -- wanted sequence length
        """
        self.paths_X = paths_X
        self.paths_t = paths_t
        self.seq_len = seq_len

        self._load_data()
        self._preprocess_data()

    def _load_data(self):
        """Read data from files and create list of samples."""

        data_X, data_t = [], []
        for path_X, path_t in zip(self.paths_X, self.paths_t):
            print("loading X data ({0})...".format(path_X))
            with open(path_X, "r", encoding="utf-8") as f:
                data_X += f.read().split("\n")
            print("loading t data ({0})...".format(path_t))
            with open(path_t, "r", encoding="utf-8") as f:
                data_t += f.read().split("\n")

        self.samples = zip(data_X, data_t)

    def _preprocess_data(self):
        """Clean up, filter, and sort the data samples before use.

        - Remove surrounding whitespace characters.
        - Filter the samples by their lengths.
        - Sort the samples for easy bucketing.
        """
        # Strip surrounding whitespace characters from each sentence
        self.samples = [(x.strip(), t.strip()) for x, t in self.samples]

        print("removing very long and very short samples ...")
        samples_before = len(self.samples)  # Count before filtering
        self.samples = _filter_samples(self.samples, float('inf'))
        self.samples = _truncate_samples(self.samples, self.seq_len-1)
        samples_after = len(self.samples)  # Count after filtering

        # Print status (number and percentage of samples left)
        samples_percentage = samples_after/samples_before*100
        subs_tuple = (samples_after, samples_before, samples_percentage)
        print("{:d} of {:d} ({:.2f}%) samples remaining".format(*subs_tuple))


class BucketIterationSchedule(frost.IterationSchedule):
    def __init__(self, shuffle=False, repeat=False, fuzzyness=3, sort=False):
        self.shuffle = shuffle
        self.repeat = repeat
        self.fuzzyness = fuzzyness
        self.sort = sort

    def gen_indices(self, loader, batch_size):
        """Yields lists of indices that make up batches.

        Make batches using the lists of indices that this function yields
        by picking the samples from the loader that have the given indices.

        Keyword arguments:
        loader -- Loader whose samples should be iterated over.
        batch_size -- Desired batch size.
        fuzzyness -- Sequence lengths within a batch will typically
            vary this much. More fuzzyness means batches will be more
            varied from one epoch to the next.
        """
        sample_indices = np.array(
            [(i, (len(x[0])//self.fuzzyness)*10000 + len(x[1])//self.fuzzyness)
             for i, x in enumerate(loader.samples)]
        )
        if self.sort:
            sample_indices = sample_indices[
                sample_indices[:, 1].argsort(kind='mergesort')]
        num_samples = len(sample_indices)
        num_batches = math.ceil(num_samples/batch_size)

        batch_numbers = range(num_batches)
        while True:
            if self.shuffle:
                # shuffling and then sorting the indices makes the batches differ between
                # epochs (unless there are so few samples that the sort operation makes
                # the samples end up in the same order no matter how they were ordered
                # before)
                np.random.shuffle(sample_indices)
                # use stable sorting algorithm so result depends on the shuffled indices.
                print('sorting samples by lengths')
                sample_indices = sample_indices[
                    sample_indices[:, 1].argsort(kind='mergesort')]
                batch_numbers = np.random.permutation(num_batches)
            for batch in batch_numbers:
                start_index = batch*batch_size
                end_index = min((batch+1)*batch_size, num_samples-1)
                yield sample_indices[start_index:end_index, 0]

            if not self.repeat:
                break


class WarmupIterationSchedule(BucketIterationSchedule):
    def __init__(self, shuffle=False, repeat=False, fuzzyness=3, sort=False,
                 warmup_iterations=10):
        self.shuffle = shuffle
        self.repeat = repeat
        self.fuzzyness = fuzzyness
        self.sort = sort
        self.warmup_iterations = warmup_iterations

    def gen_indices(self, loader, batch_size):
        # Back up attributes before changing them
        shuffle = self.shuffle
        repeat = self.repeat
        fuzzyness = self.fuzzyness
        self.shuffle = False
        self.repeat = False
        self.fuzzyness = 1
        self.sort = True
        # Do warmup
        generator = super(WarmupIterationSchedule, self).gen_indices(
            loader, batch_size)
        for i, indices in enumerate(generator):
            if i == self.warmup_iterations:
                break
            yield indices

        # Restore backed up attributes
        self.shuffle = shuffle
        self.repeat = repeat
        self.fuzzyness = fuzzyness
        self.sort = False
        # Start the real schedule
        generator = super(WarmupIterationSchedule, self).gen_indices(
            loader, batch_size)
        for indices in generator:
            yield indices


class TextBatchGenerator(frost.BatchGenerator):
    """Generates processed batches of text.

    Extends BatchGenerator
    """

    def __init__(self, loader, batch_size, seq_len, iteration_schedule=None,
            add_feature_dim=False, use_dynamic_array_sizes=False,
            alphabet=None):
        """Initialize instance of TextBatchGenerator.

        NOTE: The size of a produced batch can be smaller than batch_size if 
        there aren't enough samples left to make a full batch.

        Keyword arguments:
        loader -- instance of TextLoader.
        batch_size -- the max number of samples to include in each batch.
        seq_len -- the length of the sequences from the loader.
        iteration_schedule -- instance of IterationSchedule to be used.
            (default: BucketIterationSchedule)
        add_feature_dim -- (default: False) whether or not to add
            artificial 2nd axis to numpy arrays produced by this
            BatchGenerator.
        use_dynamic_array_sizes -- (default: False) Allow producing
            arrays of varying size along the 1st axis, to fit the
            longest sample in the batch.
        alphabet -- (optional) A custom Alphabet instance to use for encoding.
        """
        iteration_schedule = iteration_schedule or BucketIterationSchedule()
        super(TextBatchGenerator, self).__init__(loader, batch_size, iteration_schedule)

        self.alphabet = alphabet or Alphabet(eos='*', sos='')
        self.seq_len = seq_len
        self.add_feature_dim = add_feature_dim
        self.use_dynamic_array_sizes = use_dynamic_array_sizes
        self.add_eos_character = True

    def _make_batch(self):
        """Process the list of samples into a nicely formatted batch.

        Process the list of samples stored in self.samples. Return the
        result as a dict with nicely formatted numpy arrays.
        """
        encode = self.alphabet.encode
        x, t = zip(*self.samples)  # unzip samples
        batch = dict()

        batch['x_encoded'] = self._make_array(x, encode, self.seq_len)
        batch['t_encoded'] = self._make_array(t, encode, self.seq_len)
        batch['t_encoded_go'] = self._add_sos(batch['t_encoded'])

        batch['x_spaces'] = self._make_array(x, self._spaces, self.seq_len//4)
        batch['t_mask'] = self._make_array(t, self._mask, self.seq_len)

        batch['x_len'] = _make_len_vec(x, self.add_eos_character)
        batch['t_len'] = _make_len_vec(t, self.add_eos_character)

        # NOTE: The way we make batch['x_spaces_len'] here is not elegant,
        # because we compute self._spaces for the second time on the same batch
        # of samples. Think of a way to fix this!
        spaces_x = map(self._spaces, x)
        batch['x_spaces_len'] = _make_len_vec(spaces_x, 0, (self.seq_len//4))

        # Maybe add feature dimension as last part of each array shape:
        if self.add_feature_dim:
            for key, array in batch.iteritems():
                batch[key] = np.expand_dims(array, axis=-1)
        return batch

    def _make_array(self, sequences, function, max_len=None):
        """Use function to pre-process each sequence in the batch.

        Return as numpy array.

        If `self.use_dynamic_array_sizes` is True, or `max_len` has not
        been provided, the returned array's size along the 1st axis
        will be made large enough to hold the longest sequence.

        Keyword arguments:
        sequences -- list of sequences.
        function  -- the function that should be used to pre-process
            each sequence in the list before packing.
        max_len   -- (optional) size of returned array along 1st axis
            if self.use_dynamic_array_sizes is false.
        """
        if self.use_dynamic_array_sizes or not max_len:
            # make the array long enough for the longest sequence in sequences
            max_len = max([len(seq) for seq in sequences])

        array = np.zeros([self.latest_batch_size, max_len])

        # copy data into the array:
        for sample_idx, seq in enumerate(sequences):
            processed_seq = function(seq)
            length = min(max_len, len(processed_seq))
            array[sample_idx, :length] = processed_seq[:max_len]

        return array

    def _spaces(self, sentence):
        """Locate the spaces and the end of the sentence.

        Return their indices in a numpy array.
        """
        if self.add_eos_character:
            # the EOS character will be counted as a space
            sentence += " "
        spaces = [idx-1 for idx, c in enumerate(sentence) if c == " "]
        spaces.append(len(sentence)-1)
        return spaces

    def _mask(self, sentence):
        """Create a list of 1's as long as sentence."""
        mask = [1]*len(sentence)

        # Maybe add another item to the list to represent EOS character
        if self.add_eos_character:
            mask.append(1)

        return mask

    def _add_sos(self, array):
        """Add Start Of Sequence character to an array of sequences."""
        sos_col = np.ones([self.latest_batch_size, 1]) * self.alphabet.sos_id
        return np.concatenate([sos_col, array[:, :-1]], 1)


class SampleTrainWrapper(object):
    def __init__(self, loader, indexes=None, num_splits=1):

        self.loader = loader
        self.indexes = indexes
        self.num_splits = num_splits
        self.cur_split = None  # hack for batch_wrapper

        self._make_splits()
        self._make_samplers()

    def _make_splits(self):
        self.splits = []
        if self.indexes is None:
            num_samples = len(self.loader.samples)
            self.indexes = range(num_samples)
        else:
            num_samples = len(self.indexes)
        split_size = num_samples//self.num_splits
        for split in range(self.num_splits):
            self.splits.append(list(self.indexes[
                split * split_size:
                (split + 1) * split_size]))
        if num_samples - (split_size * self.num_splits):
            self.splits[-1] += list(self.indexes[
                split_size * self.num_splits:])

    def _make_samplers(self):
        self.samplers = []
        for split in self.splits:
            self.samplers.append(
                frost.SampleGenerator(self.loader, indexes=split,
                    shuffle=True, repeat=True))

    def gen_sample(self):
        while True:
            yield next(self.samplers[self.cur_split].gen_sample())
            # should add some stop mechanism.


class BatchTrainWrapper(TextBatchGenerator):

    def __init__(self, sample_generator, batch_size, seq_len,
        add_feature_dim=False, use_dynamic_array_size=False,
        alphabet=None, warm_up=5000):

        super(BatchTrainWrapper, self).__init__(sample_generator, batch_size,
            seq_len, add_feature_dim, use_dynamic_array_size, alphabet)

        self.warm_up = warm_up
        self.sample_generator.cur_split = 0

    def gen_batch(self):
        self.samples = []
        for sample in self.sample_generator.gen_sample():
            self.samples.append(sample)
            if len(self.samples) == self.batch_size:
                yield self._make_batch()
                self.samples = []
                # choose the sample_generator list
                if self.warm_up > 0:
                    self.warm_up -= 1
                    if self.warm_up == 0:
                        print('---WARM-UP OVER---')
                else:
                    self.sample_generator.cur_split = \
                        np.random.choice(self.sample_generator.num_splits)


if __name__ == '__main__':
    seq_len = 300
    text_loader = TextLoader(
        ['data/train/europarl-v7.fr-en.en'],
        ['data/train/europarl-v7.fr-en.fr'], seq_len=seq_len)
    itersched = WarmupIterationSchedule(shuffle=True)
    text_batch_gen = TextBatchGenerator(text_loader, batch_size=32, seq_len=seq_len,
                                        iteration_schedule=itersched)

    for i, batch in enumerate(text_batch_gen.gen_batch()):
        #print(batch['x_len'])
        #print(batch['t_len'])
        print(batch['x_len'][0])
        if i == 20:
            break

    print(type(batch))
    print(len(batch.items()))
    for key, item in batch.items():
        print(key, item.shape)
