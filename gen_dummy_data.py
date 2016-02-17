import numpy as np
from collections import Counter

target_to_text = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight',
    9: 'nine',
}

alphabet = Counter(''.join(target_to_text.values())).keys()
max_length = 5


def get_batch(batch_size=3):
    def encode(digit):
        return np.array(
                [alphabet.index(letter) for letter in target_to_text[digit]]
            )
    targets = np.random.choice(10, batch_size)
    encoded = [encode(target) for target in targets]
    lengths = [len(seq) for seq in encoded]

    in_mat = np.zeros([batch_size, max(lengths)], dtype=np.int)
    mask = np.zeros([batch_size, max(lengths)], dtype=np.bool)

    for i, enc in enumerate(encoded):
        in_mat[i, 0:lengths[i]] = encoded[i]
        mask[i, 0:lengths[i]] = np.ones(lengths[i])
    out = [encode(digit) for digit in targets]
    return targets, in_mat, mask

# run this file to see an example of the generated data:
if __name__ == '__main__':
    print get_batch()
    print 'alphabet size:', len(alphabet)
