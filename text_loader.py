import math
import nltk
import numpy as np
import os, subprocess

import frostings.loader as frost
from data.alphabet import Alphabet
from utils.change_directory import cd

PRINT_SEP = "  " # spaces to prepend to print statements

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

def get_npy_path(loader):
    """ From input data create file path for npy file consisting of all file
    names concatenated with _.

    NOTE: Assumes that data files are stored two folders down.
    """
    data_paths = [p.split("/") for p in loader.paths_X]
    base = data_paths[0][0] + "/" + data_paths[0][1]
    data_files = [f[2] for f in data_paths]
    data_files = "_".join(data_files)
    return base + "/" + data_files + '.npy'

def tokenize_data(samples, num_samples):
    """ Return numpy array with tokenized data """
    worded = list()
    for i, s in enumerate(samples):
        num_words = len(nltk.word_tokenize(s[0]))
        print(s[0],num_words)
        worded.append((i, num_words))
        if i % 10000 == 9999:
            pass
            #print("\r  {:.3f}% samples tokenized".format(i/num_samples*100), end="")

    return np.array(worded)

def num_word_sample_indices(loader):
    """ Return array of tuples, where the first element is the index
    and second element represents number of words in sentence.
    We have that x = s[0] and t = s[1].

    NOTE: we use nltk.word_tokenize() which splits punctuation into its own
    word. Thus we have length of individual words and punctuation.

    NOTE: if file with data exists it will be loaded, otherwise newly created
    data will be saved to disk
    """
    npy_path = get_npy_path(loader)
    if os.path.exists(npy_path):
        worded = np.load(npy_path)
    else:
        num_samples = len(loader.samples)
        print("{:s} not found -- will tokenize {:d} samples".format(
            npy_path, num_samples))
        worded = tokenize_data(loader.samples, num_samples)

        print("\n  Done -- writing tokenized array to disk.")
        np.save(npy_path, worded)
        print("  wrote data to {:s}".format(npy_path))

    return worded

def weighted_sample_indices(samples, fuzzyness):
    """ Return array of tuples, where the first element is the index
    and second element represents a weighted value used for sorting.
    We weight the length of the input sentence greater than the
    target sentence (x = s[0] and t = s[1])
    """
    return np.array(
        [(i, int((len(s[0])//fuzzyness)<<14) + len(s[1])//fuzzyness)
         for i, s in enumerate(samples)] )

def bucket_schedule(loader, batch_size, shuffle=False, repeat=False, fuzzyness=3,
    sort=False, use_word_indexer=False):
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
    use_word_indexer -- Whether or not to have `sample_indices` defining number
        of words in samples.
    """
    sample_indices = ( weighted_sample_indices(loader.samples, fuzzyness)
        if not use_word_indexer else num_word_sample_indices(loader) )

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
            with cd(data_folder): subprocess.call(['sh','./get_europarl.sh'])
            break


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
        self.samples = _filter_samples(self.samples, float('inf'))
        self.samples = _truncate_samples(self.samples, self.seq_len-1)
        samples_after = len(self.samples)  # Count after filtering

        # Print status (number and percentage of samples left)
        samples_percentage = samples_after/samples_before*100
        subs_tuple = (samples_after, samples_before, samples_percentage)
        print("{:s}{:d} of {:d} ({:.2f}%) samples remaining".format(PRINT_SEP, *subs_tuple))


class TextBatchGenerator(frost.BatchGenerator):
    """Generates processed batches of text.

    Extends BatchGenerator
    """

    def __init__(self, loader, batch_size, add_feature_dim=False,
            use_dynamic_array_sizes=False, alphabet=None, **schedule_kwargs):
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

        self.alphabet = alphabet or Alphabet(eos='*', sos='')
        self.seq_len = loader.seq_len
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
        batch['x_mask'] = self._make_array(x, self._mask, self.seq_len)
        batch['t_mask'] = self._make_array(t, self._mask, self.seq_len)

        batch['x_len'] = self._make_len_vec(x, self.add_eos_character)
        batch['t_len'] = self._make_len_vec(t, self.add_eos_character)

        # NOTE: The way we make batch['x_spaces_len'] here is not elegant,
        # because we compute self._spaces for the second time on the same batch
        # of samples. Think of a way to fix this!
        spaces_x = map(self._spaces, x)
        batch['x_spaces_len'] = self._make_len_vec(spaces_x, 0, (self.seq_len//4))

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

    def _add_sos(self, array):
        """Add Start Of Sequence character to an array of sequences."""
        sos_col = np.ones([self.latest_batch_size, 1]) * self.alphabet.sos_id
        return np.concatenate([sos_col, array[:, :-1]], 1)


class WordedTextBatchGenerator(TextBatchGenerator):
    """ Genereates processed text batches, where each element is a single word.
    Should probably only be used with dynamic RNN, because each batch element
    is not guaranteed to have same length.

    Extends TextBatchGenerator
    """

    def __init__(self, loader, batch_size, add_feature_dim=False,
            use_dynamic_array_sizes=False, alphabet=None, **schedule_kwargs):
        # this generator only works with indexing on words
        schedule_kwargs['use_word_indexer'] = True
        super(WordedTextBatchGenerator, self).__init__(loader, batch_size,
                add_feature_dim=add_feature_dim,
                use_dynamic_array_sizes=use_dynamic_array_sizes,
                alphabet=alphabet, **schedule_kwargs )

    def _make_worded_array(self, listed_sentence, encoder, max_word_len):
        """ Create arrays of arrays of words.
        Size of each sample's array will be equal to no. words for biggest
        sample. If size of sample is smaller, then we pad with zeros.
        Size of array containing word will be equal to longest word.
        """
        # [complete batch length, max word length]
        complete_batch_len = self.latest_batch_size * self.current_max_sample_size
        array = np.zeros([complete_batch_len, max_word_len], dtype='int32')

        # copy data into the array:
        for sample_idx, words in enumerate(listed_sentence):
            stride = sample_idx * self.current_max_sample_size
            for word_idx, word in enumerate(words):
                processed_word = encoder(word)
                length = min(max_word_len, len(processed_word))
                array[stride+word_idx, :length] = processed_word[:length]

        return array

    def _make_worded_len(self, listed_sentence):
        """ Returns numpy array defining length of each word in each sample.
        If sample has fewer words than longest sample, then the index will be
        zero.
        """
        result = np.array([], dtype='int32')
        for words in listed_sentence:
            word_lens = np.zeros([self.current_max_sample_size], dtype='int32')
            for i, word in enumerate(words):
                word_lens[i] = len(word)
            result = np.concatenate((result, word_lens))
        return result

    def _make_batch(self):
        """Process the list of samples into a nicely formatted batch.

        Process the list of samples stored in self.samples. Return the
        result as a dict with nicely formatted numpy arrays.

        NOTE `current_max_sample_size` defines size of longest sample in current
        batch. Use it with `batch['x_len']` to figure out if sample has ended.
        """
        encode = self.alphabet.encode
        x, t = zip(*self.samples)  # unzip samples
        batch = dict()

        words_per_x = [nltk.word_tokenize(s) for s in x]
        # define the longest sample in current batch
        self.current_max_sample_size = len(max(words_per_x, key=len))
        max_word_len = len(max([max(word, key=len) for word in words_per_x],
            key=len))
        # following two elements are different
        batch['x_encoded'] = self._make_worded_array(words_per_x, encode, max_word_len)
        batch['x_len'] = self._make_worded_len(words_per_x)

        batch['t_encoded'] = self._make_array(t, encode, self.seq_len)
        batch['t_encoded_go'] = self._add_sos(batch['t_encoded'])
        batch['x_spaces'] = self._make_array(x, self._spaces, self.seq_len//4)
        batch['t_mask'] = self._make_array(t, self._mask, self.seq_len)
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


if __name__ == '__main__':
    SEQ_LEN = 300
    BATCH_SIZE = 32
    KWARGS = { 'warmup_iterations': 20,
               'regular_function': bucket_schedule,
               'shuffle': True,
               'sort': True,
               'use_word_indexer': True }

    text_loader = TextLoader(
        ['data/train/europarl-v7.fr-en.en'],
        ['data/train/europarl-v7.fr-en.fr'], SEQ_LEN)

    #text_batch_gen = TextBatchGenerator(text_loader, BATCH_SIZE, **KWARGS)
    worded_batch_gen = WordedTextBatchGenerator(text_loader, BATCH_SIZE, **KWARGS)

    alphabet = worded_batch_gen.alphabet

    print("running warmup for 20 iterations, and 180 iterations with bucket")
    line = ""
    for i, batch in enumerate(worded_batch_gen.gen_batch(warmup_schedule)):
        line += str(batch['x_len'][0])
        line += "\n" if i % 10 == 9 else "\t"
        if i % 20 == 19:
            print(line)
            line = ""
        if i == 200:
            break

    for j in range(batch['x_len'].shape[0]):
        word = alphabet.decode(batch['x_encoded'][j])
        word_len  = batch['x_len'][j]
        print('{0} ({1})'.format(word, word_len))

    print(type(batch))
    print(len(batch.items()))
    for key, item in batch.items():
        print(key, item.shape)
