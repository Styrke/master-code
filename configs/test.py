from configs import default
from data.alphabet import Alphabet


class Model(default.Model):
    # overwrite config
    batch_size_train = 50000
    batch_size_valid = 64
    seq_len_x = 50
    seq_len_t = 50

    char_encoder_units = 200
    word_encoder_units = 200
    h1_encoder_units = 200
    embedd_dims = 64
    attn_units = 50
    attn_rnn_units = 50

    save_freq = 0
    log_freq = 100
    valid_freq = 1000  # Using a smaller valid set for debug so can do it more frequently.

    # datasets
    train_x_files = ['data/train/europarl-v7.da-en.da']
    train_t_files = ['data/train/europarl-v7.da-en.en']
    valid_x_files = ['data/valid/devtest2006.da']
    valid_t_files = ['data/valid/devtest2006.en']
    test_x_files = ['data/valid/test2007.da']
    test_t_files = ['data/valid/test2007.en']

    # settings that are local to the model
    alphabet_src = Alphabet('data/alphabet/dict_europarl.da-en.da', eos='*')
    alphabet_tar = Alphabet('data/alphabet/dict_europarl.da-en.en', eos='*', sos='')
