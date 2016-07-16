from configs.char_to_char import base
from data.alphabet import Alphabet

class Model(base.Model):
    # overwrite config
    name = 'char_to_char/europarl-ende'
    # datasets
    train_x_files = ['data/train/europarl-v7.de-en.en']
    train_t_files = ['data/train/europarl-v7.de-en.de']
    valid_x_files = ['data/valid/devtest2006.en']
    valid_t_files = ['data/valid/devtest2006.de']
    test_x_files = ['data/valid/test2008.en']
    test_t_files = ['data/valid/test2008.de']

    # settings that are local to the model
    alphabet_src = Alphabet('data/alphabet/dict_europarl.de-en.en', eos='*')
    alphabet_tar = Alphabet('data/alphabet/dict_europarl.de-en.de', eos='*', sos='')
