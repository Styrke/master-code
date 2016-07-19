from configs import default


class Model(default.Model):
    # overwrite config
    name = 'batch-size-1024'
    batch_size = 1024
