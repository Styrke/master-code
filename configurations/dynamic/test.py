import sys
sys.path.insert(0, '../..')
from configurations import dynamic_model


class Model(dynamic_model.Model):
    # overwrite config
    batch_size = 32
    seq_len = 25

    rnn_units = 100
    attn_units = 100

    valid_freq = 100  # Using a smaller valid set for debug so can do it more frequently.
    # only use a single of the validation files for this debugging config
    valid_x_files = ['data/valid/devtest2006.en']
    valid_t_files = ['data/valid/devtest2006.da']
