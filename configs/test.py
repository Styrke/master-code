from configs import default


class Model(default.Model):
    # overwrite config
    batch_size = 32
    seq_len = 25

    char_encoder_units = 100
    word_encoder_units = 100
    embedd_dims = 64

    save_freq = 0
    log_freq = 10
    valid_freq = 100  # Using a smaller valid set for debug so can do it more frequently.

    # only use a single of the validation files for this debugging config
