from frostings.loader import *
import numpy as np
import gzip
import os


def remove_samples(samples):
    # remove input sentences that are too short or too long
    samples = [(x, t) for x, t in samples if len(x) > 1 and len(x) <= 400]

    # remove target sentences that are too short or too long
    samples = [(x, t) for x, t in samples if len(t) > 1 and len(t) <= 450]

    return samples


class TextLoadMethod(LoadMethod):

    def __init__(self):
        self._prepare_data()

    def _load_data(self):
        print "loading X data ..."
        with open("data/train/europarl-v7.fr-en.en", "r") as f:
            self.train_X = f.read().split("\n")
        print "loading t data ..."
        with open("data/train/europarl-v7.fr-en.fr", "r") as f:
            self.train_t = f.read().split("\n")
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

# BATCH
# prepare a dictionary for mapping characters to integer tokens


def get_dictionary_char(lang='en'):
    with open('alphabet.' + lang, 'r') as f:
        # Convert Microsoft CRLF line endings
        alphabet_raw = f.read().replace('\r\n', '\n').replace('\r', '\n')

        # Remove duplicate entries - This shouldn't make a difference. The
        # alphabet file should only contain unique characters
        alphabet = list(set(alphabet_raw))
    return {character: idx for idx, character in enumerate(alphabet)}


def char_encoding(sentence, alphadict):
    # gets the encoding e.g. a = 180
    def encode(c):
        return alphadict[c]

    # concatenating each char in the string to
    # np.array.shape(len(sentence), len(alphadict))
    encoding = [encode(c) for c in sentence]
    return np.array(encoding)


def spaces(sentence):
    spaces = [idx-1 for idx, c in enumerate(sentence) if c == " "]
    spaces.append(len(sentence)-1)
    return np.array(spaces)


def char_length(in_string):
    return len(in_string)


class TextBatchGenerator(BatchGenerator):
    def _preprocess_sample(self):
        char_dict = dict()
        char_dict[0] = get_dictionary_char()
        char_dict[1] = get_dictionary_char('fr')
        for sample_idx, sample in enumerate(self.samples):
            my_s = []

            for elem_idx, elem in enumerate(sample):
                my_s.append(char_encoding(elem, char_dict[elem_idx]))  # code chars
                my_s.append(spaces(elem))  # spaces
                my_s.append(char_length(elem))  # char lengths

            # + sample # concats with original sample
            self.samples[sample_idx] = tuple(my_s)

    def _make_batch_holder(self, mlen_t_X, mln_s_X, mlen_t_t, mlen_s_t):
        self.batch = []
        self.batch.append(np.zeros((self.batch_info.batch_size, mlen_t_X, 1)))
        self.batch.append(np.zeros((self.batch_info.batch_size, 1)))
        self.batch.append(np.zeros((self.batch_info.batch_size, mln_s_X, 1)))
        self.batch.append(np.zeros((self.batch_info.batch_size, 1)))
        self.batch.append(np.zeros((self.batch_info.batch_size, mlen_t_t, 1)))
        self.batch.append(np.zeros((self.batch_info.batch_size, 1)))
        self.batch.append(np.zeros((self.batch_info.batch_size, mlen_s_t, 1)))
        self.batch.append(np.zeros((self.batch_info.batch_size, 1)))

        # pass # should make a "holder", e.g.
        # self.batch.append(np.zeros((self.batch_info.batch_size, max_length,
        # encoding_size) and .append a np.zeros for sequences_lengths, spaces
        # etc.

    def _make_batch(self):
        self._preprocess_sample()
        mlen_t_X = max(self.samples, key=lambda x: x[2])[2]
        mlen_s_X = len(max(self.samples, key=lambda x: len(x[1]))[1])
        mlen_t_t = max(self.samples, key=lambda x: x[5])[5]
        mlen_s_t = len(max(self.samples, key=lambda x: len(x[4]))[4])
        print mlen_t_X
        print mlen_s_X
        print mlen_t_t
        print mlen_s_t
        self._make_batch_holder(mlen_t_X, mlen_s_X, mlen_t_t, mlen_s_t)
        for sample_idx, (t_X, s_X, l_X, t_t, s_t, l_t) in enumerate(self.samples):
            t_X = np.array([t_X], dtype='float32').T  # to shape (len,1)
            l_s_X = len(s_X)
            s_X = np.array([s_X], dtype='float32').T
            t_t = np.array([t_t], dtype='float32').T
            l_s_t = len(s_t)
            s_t = np.array([s_t], dtype='float32').T
            self.batch[0][sample_idx][:l_X] = t_X
            self.batch[1][sample_idx] = l_X
            self.batch[2][sample_idx][:l_s_X] = s_X
            self.batch[3][sample_idx] = l_s_X
            self.batch[4][sample_idx][:l_t] = t_t
            self.batch[5][sample_idx] = l_t
            self.batch[6][sample_idx][:l_s_t] = s_t
            self.batch[7][sample_idx] = l_s_t

#            assert False
            #            self.batch[0][sample_idx] =

        self.samples = []  # resetting
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

    i = 0
    for batch in text_batch_gen.gen_batch():
        print i
        i += 1
        if i == 5:
            break

    print batch[0].shape
