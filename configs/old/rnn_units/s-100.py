from configs import default


class Model(default.Model):
    # overwrite config
    name = 'rnn_units/s-100'
    char_encoder_units = 100
    word_encoder_units = 100

