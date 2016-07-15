from configs.char2char import char2char

class Model(char2char.Model):
    # overwrite config
    seq_len_x = 250
    seq_len_t = 500
    valid_freq = 1000
