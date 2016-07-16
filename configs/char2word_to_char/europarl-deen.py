from configs.char2word_to_char import base
from data.alphabet import Alphabet

class Model(c2w.Model):
    # overwrite config
    name = 'char2word_to_char/europarl-deen'
    # datasets
    train_x_files = ['data/train/europarl-v7.de-en.de']
    train_t_files = ['data/train/europarl-v7.de-en.en']
    valid_x_files = ['data/valid/devtest2006.de']
    valid_t_files = ['data/valid/devtest2006.en']
    test_x_files = ['data/valid/test2007.de']
    test_t_files = ['data/valid/test2007.en']

    # settings that are local to the model
    alphabet_src = Alphabet('data/alphabet/dict_europarl.de-en.de', eos='*')
    alphabet_tar = Alphabet('data/alphabet/dict_europarl.de-en.en', eos='*', sos='')
