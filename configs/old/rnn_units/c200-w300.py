from configs import default


class Model(default.Model):
    # overwrite config
    name = 'rnn_units/c200-w300'
    char_encoder_units = 200
    word_encoder_units = 300
