from frostings.loader import *
import numpy as np
import gzip
import os

EOS = '<EOS>' # denotes end of sequence

def remove_samples(samples):
    # remove input sentences that are too short or too long
    samples = [(x, t) for x, t in samples if len(x) > 2 and len(x) <= 400]

    # Remove input sentences that that has too many spaces. This is a strict
    # inequality because we add a separater at the end of the sequence as well.
    samples = [(x, t) for x, t in samples if x.count(' ') < 65]

    # remove target sentences that are too short or too long
    samples = [(x, t) for x, t in samples if len(t) > 2 and len(t) <= 450]

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

        self.samples=[]
        for idx, (elem_X, elem_t) in enumerate(zip(self.train_X, self.train_t)): 
            self.samples.append((elem_X + EOS, elem_t + EOS))
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
                                # also addind '<EOS>'
        alphabet = list(set(alphabet_raw)) + [EOS]
    return {character: idx for idx, character in enumerate(alphabet)}


def encode(sentence, alphadict):
    return [alphadict[c] for c in sentence]


def spaces(sentence):
    spaces = [idx-1 for idx, c in enumerate(sentence) if c == " "]
    spaces.append(len(sentence)-1)
    return spaces


def char_length(in_string):
    return len(in_string)


class TextBatchGenerator(BatchGenerator):
    def __init__(self, sample_generator, batch_info, add_feature_dim=False,
                 dynamic_array_sizes=False):
        # call superclass constructor
        super(TextBatchGenerator, self).__init__(sample_generator, batch_info)

        # get alphabet dictionary for each language
        self.alphadict = dict()
        self.alphadict[0] = get_dictionary_char()
        self.alphadict[1] = get_dictionary_char('fr')

        self.add_feature_dim = add_feature_dim
        self.dynamic_array_sizes = dynamic_array_sizes

    def _preprocess_sample(self):
        for sample_idx, sample in enumerate(self.samples):
            my_s = []

            for elem_idx, elem in enumerate(sample):
                my_s.append(encode(elem, self.alphadict[elem_idx]))  # code chars
                my_s.append(spaces(elem))  # spaces (only needed for source lang)
                my_s.append(char_length(elem))  # char lengths

            # + sample # concats with original sample
            self.samples[sample_idx] = tuple(my_s)

    def _make_batch_holder(self, mlen_t_X, mlen_s_X, mlen_t_t):
        """Initiate numpy arrays for the data in the batch"""
        batch_size = self.batch_info.batch_size
        self.batch = dict()
        self.batch['x_encoded'] = np.zeros([batch_size, mlen_t_X])
        self.batch['x_len'] = np.zeros([batch_size])
        self.batch['x_spaces'] = np.zeros([batch_size, mlen_s_X])
        self.batch['x_spaces_len'] = np.zeros([batch_size])
        self.batch['t_encoded'] = np.zeros([batch_size, mlen_t_t])
        self.batch['t_len'] = np.zeros([batch_size])

    def _make_batch(self):
        self._preprocess_sample()

        mlen_s_X = len(max(self.samples, key=lambda x: len(x[1]))[1])
        if self.dynamic_array_sizes:
            # only make the arrays large enough to contain the longest sequence
            # in the batch
            mlen_t_X = max(self.samples, key=lambda x: x[2])[2]
            mlen_t_t = max(self.samples, key=lambda x: x[5])[5]
            self._make_batch_holder(mlen_t_X, mlen_s_X, mlen_t_t)
        else:
            # make maximum-size arrays whether or not its necessary
            self._make_batch_holder(400, 65, 450)

        for sample_idx, (t_X, s_X, l_X, t_t, s_t, l_t) in enumerate(self.samples):
            l_s_X = len(s_X)
            self.batch['x_encoded'][sample_idx][:l_X] = t_X
            self.batch['x_len'][sample_idx] = l_X
            self.batch['x_spaces'][sample_idx][:l_s_X] = s_X
            self.batch['x_spaces_len'][sample_idx] = l_s_X
            self.batch['t_encoded'][sample_idx][:l_t] = t_t
            self.batch['t_len'][sample_idx] = l_t

        if self.add_feature_dim:
            # add feature dimension as last part of the shape (alrojo insists)
            for key, array in self.batch.iteritems():
                self.batch[key] = np.expand_dims(array, axis=-1)

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
        break

    for key, item in batch.iteritems():
        print key, item.shape
