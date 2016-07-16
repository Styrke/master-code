from configs.char_to_char import base
from data.alphabet import Alphabet

class Model(base.Model):
    # overwrite config
    name = 'char_to_char/europarl-daen'
    # datasets
    train_x_files = ['data/train/europarl-v7.da-en.da']
    train_t_files = ['data/train/europarl-v7.da-en.en']
    valid_x_files = ['data/valid/devtest2006.da']
    valid_t_files = ['data/valid/devtest2006.en']
    test_x_files = ['data/valid/test2008.da']
    test_t_files = ['data/valid/test2008.en']

    # settings that are local to the model
    alphabet_src = Alphabet('data/alphabet/dict_europarl.da-en.da', eos='*')
    alphabet_tar = Alphabet('data/alphabet/dict_europarl.da-en.en', eos='*', sos='')
