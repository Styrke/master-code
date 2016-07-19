from configs import default


class Model(default.Model):
    # overwrite config
    name = 'embeddims/size-512'
    embedd_dims = 512
