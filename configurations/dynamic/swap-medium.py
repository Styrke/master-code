import sys
sys.path.insert(0, '../..')
from configurations import dynamic_model


class Model(dynamic_model.Model):
    # overwrite config
    name = 'dynamic/swap-medium'
    iterations = 32000*5
    seq_len = 50
    save_freq = 20000
    valid_freq = 500

    rnn_units = 400
    attn_units = 400

    swap_schedule = {
            0:     0.0,
            5000:  0.002,
            10000: 0.006,
            20000: 0.012,
            30000: 0.02,
            40000: 0.04,
            50000: 0.06,
            60000: 0.1,
            }
