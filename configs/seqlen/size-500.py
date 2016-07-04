from configs import default


class Model(default.Model):
    # overwrite config
    name = 'seqlen/size-500'
    batch_size = 64
    seq_len_x = 250
    seq_len_t = 500
    valid_freq = 3000
