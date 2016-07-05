from configs.europarl import c2w
from data.alphabet import Alphabet

class Model(c2d.Model):
    # overwrite config
    name = 'europarl/c2w-daen'
    # datasets
    train_x_files = ['data/train/europarl-v7.de-en.da']
    train_t_files = ['data/train/europarl-v7.de-en.en']
    valid_x_files = ['data/valid/devtest2006.da']
    valid_t_files = ['data/valid/devtest2006.en']
    test_x_files = ['data/valid/test2007.da']
    test_t_files = ['data/valid/test2007.en']

    # settings that are local to the model
    alphabet_src = Alphabet('data/alphabet/dict_europarl.de-en.da', eos='*')
    alphabet_tar = Alphabet('data/alphabet/dict_europarl.de-en.en', eos='*', sos='')
