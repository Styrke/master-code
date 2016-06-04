from configs import default


class Model(default.Model):
    # overwrite config
    name = 'batch-size-128'
    batch_size = 128
