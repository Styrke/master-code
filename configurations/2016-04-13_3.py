import sys,os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from configurations import separate_embeddings


class Model(separate_embeddings.Model):
    """Debugging configuration.

    This is the default configuration that is used when the training
    script is launched without naming a specific configuration file.

    This configuration uses the default model architecture but uses
    very small hyperparameters to enable quick debugging of training
    script and model architecture.
    """
    # overwrite config
   
    name = '2016-04-13_3'
    iterations = 4*32000
    rnn_units = 400
    embedd_dims = 16
