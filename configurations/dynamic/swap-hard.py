import sys
sys.path.insert(0, '../..')
from configurations import dynamic_model


class Model(dynamic_model.Model):
    # overwrite config
    name = 'dynamic/swap-hard'
    iterations = 32000*5
    seq_len = 50
    save_freq = 20000
    valid_freq = 500

    rnn_units = 400
    attn_units = 400

    swap_schedule = {
            0:     0.0,
            5000:  0.004,
            10000: 0.012,
            20000: 0.024,
            25000: 0.032,
            30000: 0.04,
            45000: 0.06,
            40000: 0.08,
            45000: 0.1,
            50000: 0.12,
            55000: 0.16,
            60000: 0.2,
            }
