from configurations import dynamic


class Model(dynamic.Model):
    # overwrite config
    name = 'dynamic/basic'
    iterations = 32000*5
    seq_len = 50
    save_freq = 20000

    rnn_units = 400
    attn_units = 400
