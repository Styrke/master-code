from collections import Counter
import pickle
import numpy as np


class Alphabet(object):
    """Easily encode and decode strings using a set of characters."""
    def __init__(self, alphabet, max_alphabet_size, eos=None, unk='', sos=None):
        """Get alphabet dict with unique integer values for each char.

        By default (if argument eos=None) no EOS character will be
        added when encoding strings. If eos is not None, each sequence
        that is encoded will be appended with an EOS char, and the value
        of the eos argument will be used to represent the EOS character
        when decoding sequences.

        Chars that aren't in the alphabet will be encoded as UNK
        character.

        Keyword arguments:
        alphabet -- List of characters, or string with path to a file
        that contains the alphabet. (default: 'data/alphabet')
        min_occurrence -- set a limit for returned list of characters 
            by minimum number of known occurrences (default: 50)
        eos -- string that will be used to represent EOS char when 
            decoding. Only specify if you want all encoded sequences 
            to be appended with EOS char (default: None)
        unk -- string that represents UNK character when decoding.
            (default: '')
        sos -- string that represents Start Of Sequence character when
            decoding. (default: None)
        """
        self.unk_char = unk  # Representation of UNK when decoding
        self.eos_char = eos  # Representation of EOS when decoding
        self.sos_char = sos  # Representation of SOS when decoding

        def reduce_dict_to_list(char_dict, max_size):
            chars = list(char_dict.copy().keys())
            freqs = list(char_dict.copy().values())
            sorted_idx = np.argsort(freqs)
            sorted_chars = [chars[ii] for ii in sorted_idx[::-1]]
            max_size = min(max_size, len(sorted_chars))
            char_list = sorted_chars[:max_size]
            return char_list

        if type(alphabet) is list:
            self.char_list = alphabet
        else:
            char_dict = pickle.load(open(alphabet, 'br'), encoding='utf-8')
            self.char_list = reduce_dict_to_list(char_dict, max_alphabet_size)

        # the list apparently contains some empty character ('') twice, so we
        # have to remove duplicates while preserving the order:
        # (http://stackoverflow.com/a/480227/118173)
        seen = set()
        seen_add = seen.add
        set_list = [x for x in self.char_list
                          if not (x in seen or seen_add(x))]
        if len(set_list) != len(self.char_list):
            raise Exception('You have dublicates in your dictionary, exiting ...')

        self.encode_dict = {char: i for i, char in enumerate(self.char_list)}
        self.decode_dict = {i: char for i, char in enumerate(self.char_list)}

        # ids for unknown and EOS characters
        self.unk_id = len(self.encode_dict)  # id for UNK char

        # be able to decode eos and sos characters if used
        if self.eos_char is not None:
            self.eos_id = len(self.decode_dict) + 1  # id for EOS char
            self.decode_dict[self.eos_id] = self.eos_char
        if self.sos_char is not None:
            self.sos_id = len(self.decode_dict) + 1  # id for SOS char
            self.decode_dict[self.sos_id] = self.sos_char

    def encode(self, string):
        """Encode a string to a sequence of integers."""
        encoded = [self.encode_dict.get(c, self.unk_id) for c in string]

        if self.eos_char:
            encoded.append(self.eos_id)

        return encoded

    def decode(self, seq):
        """Decode a sequence of integers to a string."""
        str_sequence = [self.decode_dict.get(i, self.unk_char) for i in seq]
        return ''.join(str_sequence)
