from configs import default


class Model(default.Model):
    # overwrite config
    char_encoder_units = 300
    word_encoder_units = 300

