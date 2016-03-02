from frostings.loader import *
import numpy as np
import gzip
import os

EOS = '<EOS>' # denotes end of sequence

def remove_samples(samples):
    # remove input sentences that are too short or too long
    samples = [(x, t) for x, t in samples if len(x) > 1 and len(x) <= 24]

    # Remove input sentences that that has too many spaces. This is a strict
    # inequality because we add a separater at the end of the sequence as well.
    samples = [(x, t) for x, t in samples if (x.count(' ') < 2 and
                                              t.count(' ') < 6)]

    # remove target sentences that are too short or too long
    samples = [(x, t) for x, t in samples if len(t) > 1 and len(t) <= 24]

    samples = list(set(samples))

    return samples


def strip_whitespace(sentences, whitespace=' \t\n\r'):
    """ strip whitespace from start and end of strings in given list

        Keyword arguments:
        sentences  -- the list of strings
        whitespace -- whitespace to strip (default ' \t\n\r')
    """
    return [sentence.strip(whitespace) for sentence in sentences]


class TextLoadMethod(LoadMethod):

    def __init__(self):
        self._prepare_data()

    def _load_data(self):
        print "loading X data ..."
        with open("data/train/europarl-v7.fr-en.en", "r") as f:
            self.train_X = strip_whitespace(f.read().split("\n"))
        print "loading t data ..."
        with open("data/train/europarl-v7.fr-en.fr", "r") as f:
            self.train_t = strip_whitespace(f.read().split("\n"))

        self.samples = zip(self.train_X, self.train_t)

    def _preprocess_data(self):
        print "removing very long and very short samples ..."
        self.samples = remove_samples(self.samples)

        print '%i samples left in the data set' % len(self.samples)

        print "sorting data ..."
        self.samples = sorted(self.samples,
                              key=lambda (X, t): len(X)*10000 + len(t))

    def _prepare_data(self):
        print "prepare_data started"
        self._load_data()
        self._preprocess_data()


def get_alphabet(filename='data/train/alphabet', additions=[EOS]):
    """ Return dictionary of alphabet with unique
        integer values for each element.
        Will append given list of additions to dictionary.

        Keyword arguments:
        filename  -- location of alphabet file (default 'data/train/alphabet')
        additions -- list of strings to add to alphabet (default ['<EOS>'])
    """
    with open(filename, 'r') as f:
        # Make sure only one type of line ending is present
        alphabet = f.read().replace('\r\n', '\n').replace('\r', '\n')
        # Create list of unique alphabet
        alphabet = list(set(alphabet))

    alphabet = {char: i for i, char in enumerate(alphabet)}

    # add additions of they exist
    if type(additions) is list and len(additions) > 0:
        for addition in additions:
            addition = str(addition) # stringify
            # make sure given addition is not already present
            try:
                alphabet[addition]
            except KeyError:
                alphabet[addition] = len(alphabet)
    else:
        print "WARNING -- given list of additions was invalid (list: %s, empty: %s)"\
                % (type(additions) is list, len(additions) == 0)

    return alphabet


def encode_sequence(sentence, alphabet, append_EOS=True):
    """ Return list of integer encoded sentence given by alphabet.
        Will concatenate list with encoded EOS by default.

        Keyword arguments:
        sentence    -- string to encode
        alphabet    -- dictionary over alphabet and encodings
        append_EOS  -- whether or not to append '<EOS>' to encoding (default True)
    """
    encoding = [alphabet[c] for c in sentence]

    if append_EOS:
        encoding += [alphabet[EOS]]

    return encoding


def spaces(sentence):
    spaces = [idx-1 for idx, c in enumerate(sentence) if c == " "]
    spaces.append(len(sentence)-1)
    return spaces


def char_length(in_string):
    return len(in_string)


def masking(sentence):
    return [1 for _ in sentence]


class TextBatchGenerator(BatchGenerator):
    def __init__(self, sample_generator, batch_size, add_feature_dim=False,
                 dynamic_array_sizes=False):
        # call superclass constructor
        super(TextBatchGenerator, self).__init__(sample_generator, batch_size)

        # get alphabet dictionary for each language
        self.alphabet = get_alphabet()

        self.add_feature_dim = add_feature_dim
        self.dynamic_array_sizes = dynamic_array_sizes

    def _preprocess_sample(self):
        for sample_idx, sample in enumerate(self.samples):
            my_s = []

            for elem_idx, elem in enumerate(sample):
                my_s.append(encode_sequence(elem, self.alphabet))
                # add dummy char to end that is not space such that
                # we have single dummy char that represents EOS
                elem += 'X'
                my_s.append(spaces(elem))
                my_s.append(char_length(elem))
                my_s.append(masking(elem))

            # + sample concats with original sample
            self.samples[sample_idx] = tuple(my_s)

    def _make_batch_holder(self, mlen_t_X, mlen_s_X, mlen_t_t):
        """Initiate numpy arrays for the data in the batch"""
        batch_size = self.latest_batch_size
        self.batch = dict()
        self.batch['x_encoded'] = np.zeros([batch_size, mlen_t_X])
        self.batch['x_len'] = np.zeros([batch_size])
        self.batch['x_spaces'] = np.zeros([batch_size, mlen_s_X])
        self.batch['x_spaces_len'] = np.zeros([batch_size])
        self.batch['t_encoded'] = np.zeros([batch_size, mlen_t_t])
        self.batch['t_len'] = np.zeros([batch_size])
        self.batch['t_mask'] = np.zeros([batch_size, mlen_t_t])

    def _make_batch(self):
        self._preprocess_sample()

        mlen_s_X = len(max(self.samples, key=lambda x: len(x[1]))[1])
        if self.dynamic_array_sizes:
            # only make the arrays large enough to contain the longest sequence
            # in the batch
            mlen_t_X = max(self.samples, key=lambda x: x[2])[2]
            mlen_t_t = max(self.samples, key=lambda x: x[6])[6]
            self._make_batch_holder(mlen_t_X, mlen_s_X, mlen_t_t)
        else:
            # make maximum-size arrays whether or not its necessary
            self._make_batch_holder(25, 2, 25)

        for sample_idx, (t_X, s_X, l_X, m_X, t_t, s_t, l_t, m_t) in enumerate(self.samples):
            l_s_X = len(s_X)
            self.batch['x_encoded'][sample_idx][:l_X] = t_X
            self.batch['x_len'][sample_idx] = l_X
            self.batch['x_spaces'][sample_idx][:l_s_X] = s_X
            self.batch['x_spaces_len'][sample_idx] = l_s_X
            self.batch['t_encoded'][sample_idx][:l_t] = t_t
            self.batch['t_len'][sample_idx] = l_t
            self.batch['t_mask'][sample_idx][:l_t] = m_t

        if self.add_feature_dim:
            # add feature dimension as last part of the shape (alrojo insists)
            for key, array in self.batch.iteritems():
                self.batch[key] = np.expand_dims(array, axis=-1)

        return self.batch

if __name__ == '__main__':
    text_load_method = TextLoadMethod()

    # needs to know how many samples we have, so it can make an idx for all of
    # them.
    sample_info = SampleInfo(len(text_load_method.samples))

    # generates one sample which consists of several elements sample = (elem,
    # elem, elem)
    sample_gen = SampleGenerator(text_load_method, sample_info)
    batch_info = BatchInfo(batch_size=32)

    # Generates a batch, being a tuples
    text_batch_gen = TextBatchGenerator(sample_gen, batch_info)

    for batch in text_batch_gen.gen_batch():
        pass

    for key, item in batch.iteritems():
        print key, item.shape
