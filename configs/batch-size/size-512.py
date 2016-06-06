from configs import default


class Model(default.Model):
    # overwrite config
    name = 'batch-size-512'
    batch_size = 512
