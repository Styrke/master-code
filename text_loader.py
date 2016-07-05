import math
import numpy as np
import os, subprocess

import frostings.loader as frost
from utils.change_directory import cd

PRINT_SEP = "  " # spaces to prepend to print statements

def _filter_samples(samples, max_length_x, max_length_t):
    """Filter out samples of extreme length."""
    # remove input sentences that are too short or too long
    samples = [(x, t) for x, t in samples
               if len(x) > 0 and len(x) <= max_length_x-1]
    # remove target sentences that are too short or too long
    samples = [(x, t) for x, t in samples
               if len(t) > 0 and len(t) <= max_length_t-1]
    return list(set(samples))

def _truncate_samples(samples, limit_x, limit_t):
    """Truncate long sentences."""
    return [(x[:limit_x], t[:limit_t]) for x, t in samples]

def bucket_schedule(loader, batch_size, shuffle=False, repeat=False, fuzzyness=3, sort=False):
    """Yields lists of indices that make up batches.

    Make batches using the lists of indices that this function yields
    by picking the samples from the loader that have the given indices.

    Keyword arguments:
    loader -- Loader instance with samples
    batch_size -- size of a batch
    shuffle -- shuffle array of indices
    repeat -- repeat for multiple full epochs
    fuzzyness -- Sequence lengths within a batch will typically vary this much.
        More fuzzyness means batches will be more varied from one epoch to the next.
    sort -- sort the array of indices by sequence length
    """
    sample_indices = np.array(
        # second element in the tuple represents some order for the samples
        # we weight the length of the input sentence greater than the
        # target sentence (x = s[0] and t = s[1])
        [(i, int((len(s[0])//fuzzyness)<<14) + len(s[1])//fuzzyness)
         for i, s in enumerate(loader.samples)] )

    if sort:
        sample_indices = sample_indices[
            sample_indices[:, 1].argsort(kind='mergesort') ]

    num_samples = len(sample_indices)
    batch_numbers = frost.get_number_of_batches(batch_size, num_samples)

    while True:
        if shuffle:
            # shuffling and then sorting the indices makes the batches differ between
            # epochs (unless there are so few samples that the sort operation makes
            # the samples end up in the same order no matter how they were ordered
            # before)
            np.random.shuffle(sample_indices)
            # use stable sorting algorithm so result depends on the shuffled indices.
            sample_indices = sample_indices[
                sample_indices[:, 1].argsort(kind='mergesort')
            ]
            batch_numbers = np.random.permutation(batch_numbers)
        for batch in batch_numbers:
            start, end = frost.get_start_end_indices(batch, batch_size, num_samples)
            yield sample_indices[start:end, 0]

        if not repeat:
            break


def variable_bucket_schedule(loader, threshold, fuzzyness=3):
    sample_indices = np.array(
        # second element in the tuple represents some order for the samples
        # we weight the length of the input sentence greater than the
        # target sentence (x = s[0] and t = s[1])
        [(i, int((len(s[0])//fuzzyness) << 14) + len(s[1])//fuzzyness, len(s[0]),
          len(s[1]))
         for i, s in enumerate(loader.samples)])

    while True:
        np.random.shuffle(sample_indices)
        sample_indices = sample_indices[
            sample_indices[:, 1].argsort(kind='mergesort')]

        batches = []
        current_batch = []
        longest_x = 0
        longest_t = 0
        for idx in sample_indices:
            longest_x = max(longest_x, idx[2])
            longest_t = max(longest_t, idx[3])
            if (longest_x+longest_t)*(len(current_batch)+1) <= threshold:
                current_batch.append(idx[0])
            else:
                batches.append(current_batch)
                current_batch = [idx[0]]
                longest_x = idx[2]
                longest_t = idx[3]

        batch_numbers = np.random.permutation(len(batches))
        for idx in batch_numbers:
            yield batches[idx]

def warmup_schedule(loader, batch_size, warmup_iterations=10, warmup_function=None,
        regular_function=None, **kwargs):
    """ Yields lists of indices that make up batches.

    We do not wish to shuffle nor repeat the schedule, and the samples are to be sorted.
    When warmup is done, we continue with regular schedule.

    Keyword arguments:
    loader -- Loader instance with samples
    batch_size -- size of a batch
    warmup_iterations -- number of iterations to run this schedule
    warmup_function -- (default: bucket_schedule) function to use for warmup
    regular_function -- (default: bucket_schedule) schedule function to be called after
        warmup has finished
    **kwargs -- arguments to be used after warmup schedule has finished
    """
    warmup_kwargs = {'shuffle': False, 'repeat': False, 'fuzzyness': 1, 'sort': True}
    warmup_function = warmup_function or bucket_schedule
    regular_function = regular_function or bucket_schedule

    # do warmup schedule
    generator = warmup_function(loader, batch_size, **warmup_kwargs)
    for i, indices in enumerate(generator):
        if i == warmup_iterations:
            break
        yield indices

    # do regular schedule
    generator = regular_function(loader, batch_size, **kwargs)
    for i, indices in enumerate(generator):
        yield indices

def validate_data_existence_or_fetch(paths, data_folder="data/"):
    """ Check that all files exist, otherwise download everything """
    for path in paths:
        if not os.path.exists(path):
            print("{0}Missing data. Will fetch everything..".format(PRINT_SEP))
            subprocess.call(['sh','data/get_data.sh'])
            break


class TextLoader(frost.Loader):
    """Load and prepare text data."""

    def __init__(self, paths_X, paths_t, seq_len_x, seq_len_t):
        """ Initialize TextLoader instance.

        Keyword arguments:
        paths_X -- list of paths for input sentences
        paths_t -- list of paths for target sentences
        seq_len -- wanted sequence length
        """
        self.paths_X = paths_X
        self.paths_t = paths_t
        self.seq_len_x = seq_len_x
        self.seq_len_t = seq_len_t

        validate_data_existence_or_fetch(paths_X + paths_t)

        self._load_data()
        self._preprocess_data()

    def _load_data(self):
        """Read data from files and create list of samples."""

        data_X, data_t = [], []
        for path_X, path_t in zip(self.paths_X, self.paths_t):
            print("{0}loading X data ({1})...".format(PRINT_SEP, path_X))
            with open(path_X, "r", encoding="utf-8") as f:
                data_X += f.read().split("\n")
            print("{0}loading t data ({1})...".format(PRINT_SEP, path_t))
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

        print("{0}removing very long and very short samples ...".format(PRINT_SEP))
        samples_before = len(self.samples)  # Count before filtering
        self.samples = _filter_samples(self.samples, max_length_x=251, max_length_t=501)
        self.samples = _truncate_samples(self.samples, limit_x=self.seq_len_x-1, limit_t=self.seq_len_t-1)
        samples_after = len(self.samples)  # Count after filtering

        # Print status (number and percentage of samples left)
        samples_percentage = samples_after/samples_before*100
        subs_tuple = (samples_after, samples_before, samples_percentage)
        print("{:s}{:d} of {:d} ({:.2f}%) samples remaining".format(PRINT_SEP, *subs_tuple))


class TextBatchGenerator(frost.BatchGenerator):
    """Generates processed batches of text.

    Extends BatchGenerator
    """

    def __init__(self, loader, batch_size, alphabet_src, alphabet_tar, add_feature_dim=False,
            use_dynamic_array_sizes=False,  **schedule_kwargs):
        """Initialize instance of TextBatchGenerator.

        NOTE: The size of a produced batch can be smaller than batch_size if
        there aren't enough samples left to make a full batch.

        Keyword arguments:
        loader -- instance of TextLoader.
        batch_size -- the max number of samples to include in each batch.
        add_feature_dim -- (default: False) whether or not to add
            artificial 2nd axis to numpy arrays produced by this
            BatchGenerator.
        use_dynamic_array_sizes -- (default: False) Allow producing
            arrays of varying size along the 1st axis, to fit the
            longest sample in the batch.
        alphabet -- (optional) A custom Alphabet instance to use for encoding.
        schedule_kwargs -- additional arguments for schedule function in
            BatchGenerator's gen_batch()
        """
        super(TextBatchGenerator, self).__init__(loader, batch_size,
                **schedule_kwargs)

        self.alphabet_src = alphabet_src
        self.alphabet_tar = alphabet_tar
        self.seq_len_x = loader.seq_len_x
        self.seq_len_t = loader.seq_len_t
        self.add_feature_dim = add_feature_dim
        self.use_dynamic_array_sizes = use_dynamic_array_sizes
        self.add_eos_character = True

    def _make_batch(self):
        """Process the list of samples into a nicely formatted batch.

        Process the list of samples stored in self.samples. Return the
        result as a dict with nicely formatted numpy arrays.
        """
        encode_src = self.alphabet_src.encode
        encode_tar = self.alphabet_tar.encode
        x, t = zip(*self.samples)  # unzip samples
        batch = dict()

        batch['x_encoded'] = self._make_array(x, encode_src, self.seq_len_x)
        batch['t_encoded'] = self._make_array(t, encode_tar, self.seq_len_t)
        batch['t_encoded_go'] = self._add_sos(batch['t_encoded'], self.alphabet_tar)

        batch['x_spaces'] = self._make_array(x, self._spaces, self.seq_len_x//4)
        batch['x_mask'] = self._make_array(x, self._mask, self.seq_len_x)
        batch['t_mask'] = self._make_array(t, self._mask, self.seq_len_t)

        batch['x_len'] = self._make_len_vec(x, self.add_eos_character)
        batch['t_len'] = self._make_len_vec(t, self.add_eos_character)

        # NOTE: The way we make batch['x_spaces_len'] here is not elegant,
        # because we compute self._spaces for the second time on the same batch
        # of samples. Think of a way to fix this!
        spaces_x = map(self._spaces, x)
        batch['x_spaces_len'] = self._make_len_vec(spaces_x, 0, (self.seq_len_x//4))

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

        sequences = list(map(function, sequences))

        if self.use_dynamic_array_sizes or not max_len:
            # make the array long enough for the longest sequence in sequences
            max_len = max([len(seq) for seq in sequences])

        array = np.zeros([self.latest_batch_size, max_len])

        # copy data into the array:
        for sample_idx, seq in enumerate(sequences):
            length = min(max_len, len(seq))
            array[sample_idx, :length] = seq[:max_len]

        return array

    def _make_len_vec(self, sequences, offset=0, max_len=100000):
        """Get length of each sequence in list of sequences.

        Return as numpy vector.

        Keyword arguments:
        sequences -- list of sequences.
        offset -- (optional) value to be added to each length.
            Useful for including EOS characters.
        max_len -- (optional) (default:100,000) returned sequence length will
            not exceed this value.
        """
        if self.use_dynamic_array_sizes:
            max_len = 100000
        return np.array([min(len(seq)+offset, max_len) for seq in sequences])

    def _spaces(self, sentence):
        """Locate the spaces and the end of the sentence.

        Return their indices in a numpy array.

        NOTE: We append the last index of the `spaces` array. The purpose is
        that if we add eos, then the last index will represent the eos
        symbol's index. Otherwise, it will give the index of the end of the
        last word.
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
        if self.add_eos_character: mask.append(1)
        return mask

    def _add_sos(self, array, alphabet):
        """Add Start Of Sequence character to an array of sequences."""
        sos_col = np.ones([self.latest_batch_size, 1]) * alphabet.sos_id
        return np.concatenate([sos_col, array[:, :-1]], 1)


if __name__ == '__main__':
    from data.alphabet import Alphabet
    SEQ_LEN_X = 250
    SEQ_LEN_T = 500
    BATCH_SIZE = 76800

    text_loader = TextLoader(
        ['data/train/europarl-v7.de-en.en'],
        ['data/train/europarl-v7.de-en.de'], SEQ_LEN_X, SEQ_LEN_T)

    alphabet_src = Alphabet('data/alphabet/dict_wmt_tok.de-en.en', eos='*')
    alphabet_tar = Alphabet('data/alphabet/dict_wmt_tok.de-en.de', eos='*', sos='')

    text_batch_gen = TextBatchGenerator(text_loader,
                                        BATCH_SIZE,
                                        alphabet_src,
                                        alphabet_tar,
                                        use_dynamic_array_sizes=True)

    print("running warmup for 20 iterations, and 180 iterations with bucket")
    line = ""
    for i, batch in enumerate(text_batch_gen.gen_batch(variable_bucket_schedule)):
        print(batch["x_encoded"].shape, batch["t_encoded"].shape)
        if i == 200:
            break

    print(type(batch))
    print(len(batch.items()))
    for key, item in batch.items():
        print(key, item.shape)
