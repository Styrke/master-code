from configs import default


class Model(default.Model):
    # overwrite config
    name = 'seqlen/size-300'
    seq_len = 300
    valid_freq = 1500
