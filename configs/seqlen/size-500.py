from configs import default


class Model(default.Model):
    # overwrite config
    name = 'seqlen/size-500'
    batch_size = 64
    seq_len = 500
    valid_freq = 3000
