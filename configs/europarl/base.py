from configs import default
from data.alphabet import Alphabet

class Model(default.Model):
    # overwrite config
    seq_len_x = 250
    seq_len_t = 500
    valid_freq = 1000
