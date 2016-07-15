from configs.europarl import c2w
from data.alphabet import Alphabet

class Model(c2w.Model):
    # overwrite config
    name = 'europarl/c2w_wmt-deen'
