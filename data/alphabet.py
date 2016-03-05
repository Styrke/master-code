from collections import Counter


class Alphabet(object):
    """Easily encode and decode strings using a set of characters."""
    def __init__(self, alphabet_file='data/alphabet', eos=None, unk=''):
        """Get alphabet dict with unique integer values for each char.

        By default (if argument eos==None) no EOS character will be
        added when encoding strings. If eos is not None, each sequence
        that is encoded will be appended with an EOS char, and the value
        of the eos argument will be used to represent the EOS character
        when decoding sequences.

        Chars that arent in the alphabet will be encoded as UNK
        character.

        Keyword arguments:
        alphabet_file -- File that contains the alphabet. (default:
            'alphabet')
        eos -- (optional) string that will be used to represent EOS
            char when decoding. Only specify if you want all encoded
            sequences to be appended with EOS char
        unk -- string that represents UNK character when decoding.
            (default: '')
        """
        self.unk_char = unk  # Representation of UNK when decoding
        self.eos_char = eos  # Representation of EOS when decoding

        with open(alphabet_file, 'r', encoding="utf-8") as f:
            self.char_list = f.read().split('\n')

        # the list apparently contains some empty character ('') twice, so we
        # have to remove duplicates while preserving the order:
        # (http://stackoverflow.com/a/480227/118173)
        seen = set()
        seen_add = seen.add
        self.char_list = [x for x in self.char_list
                          if not (x in seen or seen_add(x))]

        self.encode_dict = {char: i for i, char in enumerate(self.char_list)}
        self.decode_dict = {i: char for i, char in enumerate(self.char_list)}

        # ids for unknown and EOS characters
        self.unk_id = len(self.encode_dict)  # id for UNK char
        self.eos_id = self.unk_id + 1  # id for EOS char

        # be able to decode eos character if used
        if self.eos_char:
            self.decode_dict[self.eos_id] = self.eos_char

    def encode(self, string):
        """Encode a string to a sequence of integers."""
        encoded = [self.encode_dict.get(c, self.unk_id) for c in string]

        if self.eos_char:
            encoded.append(self.eos_id)

        return encoded

    def decode(self, sequence):
        """Decode a sequence of integers to a string."""
        str_sequence = [self.decode_dict.get(i, self.unk_id) for i in sequence]
        return ''.join(str_sequence)