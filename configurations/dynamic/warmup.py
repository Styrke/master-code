import sys
sys.path.insert(0, '../..')
from configurations import dynamic_model


class Model(dynamic_model.Model):
    # overwrite config
    name = 'dynamic/warmup'
    iterations = 32000*5
    seq_len = 50
    save_freq = 20000
    valid_freq = 500

    rnn_units = 400
    attn_units = 400

    schedule_kwargs = {
            'warmup_iterations': 10000,  # if warmup_schedule is used
            'warmup_function':  None,  # if warmup_schedule is used
            'regular_function': None,  # if warmup_schedule is used
            'shuffle': True,
            'repeat':  True,
            'sort':    False,
            'fuzzyness': 3
            }
