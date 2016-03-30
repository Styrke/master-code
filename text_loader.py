from frostings import loader
import numpy as np

from data.alphabet import Alphabet


class TextLoadMethod(loader.LoadMethod):
    """Load and prepare text data."""

    def __init__(self, paths_X, paths_t, seq_len):
        """Initialize instance of TextLoadMethod."""
        self.paths_X = paths_X
        self.paths_t = paths_t
        self.seq_len = seq_len
        self._prepare_data()

    def _load_data(self):
        """Read data from files and create list of samples."""

        data_X = []
        data_t = []
        for path_X, path_t in zip(self.paths_X, self.paths_t):
            print("loading X data ...")
            with open(path_X, "r", encoding="utf-8") as f:
                data_X += f.read().split("\n")
            print("loading t data ...")
            with open(path_t, "r", encoding="utf-8") as f:
                data_t = f.read().split("\n")

        self.samples = zip(data_X, data_t)

    def _preprocess_data(self):
        """Clean up, filter, and sort the data samples before use.

        - Remove sorrounding whitespace characters.
        - Filter the samples by their lengths.
        - Sort the samples for easy bucketing.
        """
        # Strip sorrounding whitespace characters from each sentence
        self.samples = [(x.strip(), t.strip()) for x, t in self.samples]

        print("removing very long and very short samples ...")
        self.samples = self._filter_samples(self.samples, float('inf'))
        self.samples = self._truncate_samples(self.samples)

        print("%i samples left in the data set" % len(self.samples))

        print("sorting data ...")
        self.samples = sorted(self.samples,
                              key=lambda x: len(x[0])*10000 + len(x[1]))

    def _prepare_data(self):
        """Load and preprocess data."""
        print("prepare_data started")
        self._load_data()
        self._preprocess_data()

    def _filter_samples(self, samples, max_length):
        """Filter out samples of extreme length."""
        # remove input sentences that are too short or too long
        samples = [(x, t) for x, t in samples
                   if len(x) > 1 and len(x) <= max_length-1]

        # remove target sentences that are too short or too long
        samples = [(x, t) for x, t in samples
                   if len(t) > 1 and len(t) <= max_length-1]

        samples = list(set(samples))

        return samples

    def _truncate_samples(self, samples):
        """Truncate long sentences."""
        end = self.seq_len - 1
        samples = [(x[:end], t[:end]) for x, t in samples]
        return samples



class TextBatchGenerator(loader.BatchGenerator):
    """Generates processed batches of text.

    Extends BatchGenerator
    """

    def __init__(self, sample_generator, batch_size, seq_len,
            add_feature_dim=False, use_dynamic_array_sizes=False,
            alphabet=None):
        """Initialize instance of TextBatchGenerator.

        NOTE: The size of a produced batch can be smaller than
        batch_size if there aren't enough samples left to make a full
        batch.

        Keyword arguments:
        sample_generator -- instance of SampleGenerator that has been
            initialized with a TextLoadMethod instance.
        batch_size -- the max number of samples to include in each
            batch.
        add_feature_dim -- (default: False) whether or not to add
            artificial 2nd axis to numpy arrays produced by this
            BatchGenerator.
        use_dynamic_array_sizes -- (default: False) Allow producing
            arrays of varying size along the 1st axis, to fit the
            longest sample in the batch.
        alphabet -- (optional) A custom Alphabet instance to use for
            encoding.
        """
        # call superclass constructor
        super(TextBatchGenerator, self).__init__(sample_generator, batch_size)

        if alphabet:
            self.alphabet = alphabet
        else:
            self.alphabet = Alphabet(eos='*', sos='')

        self.seq_len = seq_len
        self.add_feature_dim = add_feature_dim
        self.use_dynamic_array_sizes = use_dynamic_array_sizes
        self.add_eos_character = True

    def _make_batch(self):
        """Process the list of samples into a nicely formatted batch.

        Process the list of samples stored in self.samples. Return the
        result as a dict with nicely formatted numpy arrays.
        """
        x, t = zip(*self.samples)  # unzip samples
        self.batch = dict()
        self.batch['x_encoded'] = self._make_array(x, self.alphabet.encode,
            self.seq_len)
        self.batch['t_encoded'] = self._make_array(t, self.alphabet.encode,
            self.seq_len)
        self.batch['x_spaces'] = self._make_array(x, self._spaces,
            self.seq_len/2)
        self.batch['t_mask'] = self._make_array(t, self._mask, self.seq_len)

        self.batch['t_encoded_go'] = self._add_sos(self.batch['t_encoded'])

        offset = self.add_eos_character  # Maybe count EOS character
        self.batch['x_len'] = self._make_len_vec(x, offset=offset)
        self.batch['t_len'] = self._make_len_vec(t, offset=offset)
        self.batch['x_spaces_len'] = self._make_len_vec(map(self._spaces, x))
        # NOTE: The way we make self.batch['x_spaces_len'] here is not elegant,
        # because we compute self._spaces for the second time on the same batch
        # of samples. Think of a way to fix this!

        # Maybe add feature dimension as last part of each array shape:
        if self.add_feature_dim:
            for key, array in self.batch.iteritems():
                self.batch[key] = np.expand_dims(array, axis=-1)

        return self.batch

    def _make_array(self, sequences, function, max_len=None):
        """Use function to preprocess each sequence in the batch.

        Return as numpy array.

        If self.use_dynamic_array_sizes is true, or max_len has not
        been provided, the returned array's size along the 1st axis
        will be made large enough to hold the longest sequence.

        Keyword arguments:
        sequences -- list of sequences.
        function  -- the function that should be used to preprocess
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
            array[sample_idx, :len(processed_seq)] = processed_seq

        return array

    def _make_len_vec(self, sequences, offset=0):
        """Get length of each sequence in list of sequences.

        Return as numpy vector.

        Keyword arguments:
        sequences -- list of sequences.
        offset    -- (optional) value to be added to each length.
            Useful for including EOS characters.
        """
        return np.array([len(seq)+offset for seq in sequences])

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
    def __init__(self, load_method, permutation = None, num_splits=1):

        self.load_method = load_method
        self.permutation = permutation
        self.num_splits = num_splits
        self.cur_split = None # hack for batch_wrapper

        self._make_splits()
        self._make_samplers()

    def _make_splits(self):
        self.splits = []
        if self.permutation is None:
            num_samples = len(self.load_method.samples)
            self.permutation = range(num_samples)
        else:
            num_samples = len(self.permutation)
        split_size = num_samples//self.num_splits
        for split in range(self.num_splits):
            self.splits.append(list(self.permutation[
                split * split_size:
                (split + 1) * split_size]))
        if num_samples - (split_size * self.num_splits):
            self.splits[-1] += list(self.permutation[
                split_size * self.num_splits:])

    def _make_samplers(self):
        self.samplers = []
        for split in self.splits:
            self.samplers.append(
                loader.SampleGenerator(self.load_method, permutation=split,
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
    text_load_method = TextLoadMethod(
        ['data/train/europarl-v7.fr-en.en'],
        ['data/train/europarl-v7.fr-en.fr'], seq_len=40)
    sample_gen = loader.SampleGenerator(text_load_method)
    text_batch_gen = TextBatchGenerator(sample_gen, batch_size=32, seq_len=40)

    for batch in text_batch_gen.gen_batch():
        pass
    print(type(batch))
    print(len(batch.items()))
    for key, item in batch.items():
        print(key, item.shape)

    sample_wrapper = SampleTrainWrapper(text_load_method, num_splits=16)
    batch_wrapper = BatchTrainWrapper(sample_wrapper,
        batch_size=32, seq_len=40, warm_up=5000)
    batch = next(batch_wrapper.gen_batch())
    print(type(batch))
    print(len(batch.items()))

    for key, item in batch.items():
        print(key, item.shape)
