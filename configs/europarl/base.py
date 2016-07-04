from configs import default
from data.alphabet import Alphabet

class Model(default.Model):
    # overwrite config
    batch_size = 128
    seq_len_x = 250
    seq_len_t = 500
    valid_freq = 3000
