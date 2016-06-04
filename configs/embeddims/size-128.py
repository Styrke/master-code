from configs import default


class Model(default.Model):
    # overwrite config
    name = 'embeddims/size-128'
    embedd_dims = 128
