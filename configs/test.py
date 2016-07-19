from configs import default


class Model(default.Model):
    # overwrite config
    batch_size_train = 10000
    batch_size_valid = 64
    seq_len_x = 50
    seq_len_t = 50

    char_encoder_units = 100
    word_encoder_units = 100
    embedd_dims = 64
    attn_units = 50

    save_freq = 0
    log_freq = 10
    valid_freq = 100  # Using a smaller valid set for debug so can do it more frequently.

    # only use a single of the validation files for this debugging config
