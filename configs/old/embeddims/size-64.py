from configs import default


class Model(default.Model):
    # overwrite config
    name = 'embeddims/size-64'
    embedd_dims = 64
