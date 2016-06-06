from configs import default


class Model(default.Model):
    # overwrite config
    name = 'batch-size-256'
    batch_size = 256
