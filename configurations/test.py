import sys
sys.path.insert(0, '..')
import model


class Model(model.Model):
    # overwrite config
    batch_size = 32
