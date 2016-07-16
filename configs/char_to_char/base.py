from configs.char_to_char import char_to_char

class Model(char_to_char.Model):
    # overwrite config
    seq_len_x = 250
    seq_len_t = 500
    valid_freq = 1000
