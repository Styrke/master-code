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
    swap_schedule = {
        0: 0.0,
        5000: 0.05,
        10000: 0.1,
        20000: 0.15,
        30000: 0.25,
        40000: 0.3,
        50000: 0.35,
        60000: 0.39,
    }

   
    name = '2016-04-13_1'
    iterations = 4*32000
    rnn_units = 512
    embedd_dims = 32
