from configs import default


class Model(default.Model):
    # overwrite config
    name = 'rnn_units/c300-w300'
    char_encoder_units = 300
    word_encoder_units = 300
