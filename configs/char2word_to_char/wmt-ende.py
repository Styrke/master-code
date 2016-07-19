from configs.char2word_to_char import base
from data.alphabet import Alphabet

class Model(base.Model):
    # overwrite config
    name = 'char2word_to_char/wmt-ende'
