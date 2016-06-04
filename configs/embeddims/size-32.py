from configs import default


class Model(default.Model):
    # overwrite config
    name = 'embeddims/size-32'
    embedd_dims = 32
