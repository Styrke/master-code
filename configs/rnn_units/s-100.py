from configs import default


class Model(default.Model):
    # overwrite config
    char_encoder_units = 100
    word_encoder_units = 100

