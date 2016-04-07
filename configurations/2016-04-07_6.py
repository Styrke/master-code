import sys
sys.path.insert(0, '..')
import model


class Model(model.Model):
    """Debugging configuration.

    This is the default configuration that is used when the training
    script is launched without naming a specific configuration file.

    This configuration uses the default model architecture but uses
    very small hyperparameters to enable quick debugging of training
    script and model architecture.
    """
    # overwrite config
    name = '2016-04-07_6'
    iterations = 5*32000
    train_feedback = True
    rnn_units = 1000
    embedd_dims = 64
