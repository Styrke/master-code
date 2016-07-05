from configs import default


class Model(default.Model):
    # overwrite config
    name = 'seqlen/size-100'
    seq_len_x = seq_len_t = 100
