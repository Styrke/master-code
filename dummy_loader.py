from frostings.loader import *
import numpy as np

def simple_dummy_sample(max_len, max_len_spaces):
    # setting variables
    elem_len = np.random.choice(max_len) # generates a random len seq
    max_spaces = max_len_spaces

    # setting holders
    space_counter = 0
    elem_X = ""
    elem_t = ""

    # generating output
    for _ in range(elem_len):
        elem_X += str(np.random.choice(10))
        rand = np.random.choice(1)
        if np.random.choice(max(int(elem_len/max_spaces), 1)) == 1:
            if space_counter < max_spaces:
                elem_X += " "
                space_counter += 1
    elem_t = elem_X # X and t are the same (why it's called "simple")
    return elem_X, elem_t


class DummySampleGenerator(SampleGenerator):
    """Generates a sample from a dummy sampler

    Extends SampleGenerator
    """


    def __init__(self, make_dummy_sample=simple_dummy_sample,
                 max_len=10, max_len_spaces=1):
        self.make_dummy_sample = make_dummy_sample
        self.max_len = max_len
        self.max_len_spaces = max_len_spaces


    def gen_sample(self):
        while True:
            yield self.make_dummy_sample(self.max_len, self.max_len_spaces)

if __name__ == '__main__':
    dummy_sample_gen = DummySampleGenerator()
    count = 0
    for sample in dummy_sample_gen.gen_sample():
        print(sample)
        count += 1
        if count > 5:
            break
