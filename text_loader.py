from frostings.loader import *
import numpy as np

EOS = '<EOS>'  # denotes end of sequence


class TextLoadMethod(LoadMethod):
    """Load and prepare text data."""

    def __init__(self):
        """Initialize instance of TextLoadMethod."""
        self._prepare_data()

    def _load_data(self):
        """Read data from files and create list of samples."""
        print("loading X data ...")
        with open("data/train/europarl-v7.fr-en.en", "r") as f:
            train_X = f.read().split("\n")
        print("loading t data ...")
        with open("data/train/europarl-v7.fr-en.fr", "r") as f:
            train_t = f.read().split("\n")

        self.samples = zip(train_X, train_t)

    def _preprocess_data(self):
        """Clean up, filter, and sort the data samples before use.

        - Remove sorrounding whitespace characters.
        - Filter the samples by their lengths.
        - Sort the samples for easy bucketing.
        """
        # Strip sorrounding whitespace characters from each sentence
        self.samples = [(x.strip(), t.strip()) for x, t in self.samples]

        print("removing very long and very short samples ...")
        self.samples = self._filter_samples(self.samples)

        print("%i samples left in the data set" % len(self.samples))

        print("sorting data ...")
        self.samples = sorted(self.samples,
                              key=lambda x: len(x[0])*10000 + len(x[1]))

    def _prepare_data(self):
        """Load and preprocess data."""
        print("prepare_data started")
        self._load_data()
        self._preprocess_data()

    def _filter_samples(self, samples):
        """Filter out samples of extreme length."""
        # remove input sentences that are too short or too long
        samples = [(x, t) for x, t in samples if len(x) > 1 and len(x) <= 24]

        # Remove input sentences that that has too many spaces. This is a
        # strict inequality because we add a separater at the end of the
        # sequence as well.
        samples = [(x, t) for x, t in samples if (x.count(' ') < 2 and
                                                  t.count(' ') < 6)]

        # remove target sentences that are too short or too long
        samples = [(x, t) for x, t in samples if len(t) > 1 and len(t) <= 24]

        samples = list(set(samples))

        return samples


class TextBatchGenerator(BatchGenerator):
    """Generates processed batches of text.

    Extends BatchGenerator
    """

    def __init__(self, sample_generator, batch_size, add_feature_dim=False,
                 use_dynamic_array_sizes=False):
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
        """
        # call superclass constructor
        super(TextBatchGenerator, self).__init__(sample_generator, batch_size)

        self.alphabet = self.get_alphabet()  # get alphabet dictionary

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
        self.batch['x_encoded'] = self._make_array(x, self._encode, 25)
        self.batch['t_encoded'] = self._make_array(t, self._encode, 25)
        self.batch['x_spaces'] = self._make_array(x, self._spaces, 3)
        self.batch['t_mask'] = self._make_array(t, self._mask, 25)

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

    def _encode(self, sentence):
        """Encode sentence as a list of integers given by alphabet."""
        encoding = [self.alphabet[c] for c in sentence]

        if self.add_eos_character:
            encoding.append(self.alphabet[EOS])

        return encoding

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

    def get_alphabet(self, filename='data/train/alphabet', additions=[EOS]):
        """Get alphabet dict with unique integer values for each char.

        Will append given list of additions to dictionary.

        Keyword arguments:
        filename  -- location of alphabet file (default:
            'data/train/alphabet')
        additions -- list of strings to add to alphabet (default:
            ['<EOS>'])
        """
        with open(filename, 'r') as f:
            alphabet = f.read().split('\n')

        alphabet = {char: i for i, char in enumerate(alphabet)}

        # add any additions that aren't already in the alphabet dictionary
        for addition in additions:
            if addition not in alphabet:
                alphabet[addition] = len(alphabet)

        return alphabet


if __name__ == '__main__':
    text_load_method = TextLoadMethod()
    sample_gen = SampleGenerator(text_load_method)
    text_batch_gen = TextBatchGenerator(sample_gen, batch_size=32)

    for batch in text_batch_gen.gen_batch():
        pass

    for key, item in batch.items():
        print(key, item.shape)
