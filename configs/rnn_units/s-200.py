from configs import default


class Model(default.Model):
    # overwrite config
    name = 'rnn_units/s-200'
    char_encoder_units = 200
    word_encoder_units = 200
