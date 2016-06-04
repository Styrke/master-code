from configs import default


class Model(default.Model):
    # overwrite config
    name = 'seqlen/size-100'
    seq_len = 100
