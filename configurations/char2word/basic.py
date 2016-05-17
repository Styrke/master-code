import sys
sys.path.insert(0, '../..')
from configurations import char2word_model


class Model(char2word_model.Model):
    # overwrite config
    name = 'c2w/basic'
    iterations = 32000*5
    seq_len = 50
    save_freq = 20000
    valid_freq = 500

    char_enc_units = 400
    word_enc_units = 400
    dec_units = 400
