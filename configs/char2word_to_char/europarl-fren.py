from configs.char2word_to_char import base
from data.alphabet import Alphabet

class Model(base.Model):
    # overwrite config
    name = 'char2word_to_char/europarl-fren'
    # datasets
    train_x_files = ['data/train/europarl-v7.fr-en.fr']
    train_t_files = ['data/train/europarl-v7.fr-en.en']
    valid_x_files = ['data/valid/devtest2006.fr']
    valid_t_files = ['data/valid/devtest2006.en']
    test_x_files = ['data/valid/test2007.fr']
    test_t_files = ['data/valid/test2007.en']

    # settings that are local to the model
    alphabet_src = Alphabet('data/alphabet/dict_europarl.fr-en.fr', eos='*')
    alphabet_tar = Alphabet('data/alphabet/dict_europarl.fr-en.en', eos='*', sos='')
