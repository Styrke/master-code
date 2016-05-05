import sys
import tensorflow as tf
sys.path.insert(0, '..')
import model
from utils.tfextensions import sequence_loss_tensor

class Model(model.Model):
    """Debugging configuration.

    This is the default configuration that is used when the training
    script is launched without naming a specific configuration file.

    This configuration uses the default model architecture but uses
    very small hyperparameters to enable quick debugging of training
    script and model architecture.
    """
    # overwrite config
    batch_size = 32
    seq_len = 25
    rnn_units = 100
    valid_freq = 100  # Using a smaller valid set for debug so can do it more frequently.

    # only use a single of the validation files for this debugging config
    valid_x_files = ['data/valid/devtest2006.en']
    valid_t_files = ['data/valid/devtest2006.da']

    def build_loss(self):
        """Build a loss function and accuracy for the model."""
        print('  Building loss and accuracy')

        with tf.variable_scope('accuracy'):
            argmax = tf.to_int32(tf.argmax(self.out_tensor, 2))
            correct = tf.to_float(tf.equal(argmax, self.ts)) * self.t_mask
            self.accuracy = tf.reduce_sum(correct) / tf.reduce_sum(self.t_mask)

            tf.scalar_summary('train/accuracy', self.accuracy)

        with tf.variable_scope('loss'):
            with tf.variable_scope('split_t_and_mask'):
                split_kwargs = { 'split_dim': 1,
                                 'num_split': self.max_t_seq_len }
                ts     = tf.split(value=self.ts,     **split_kwargs)
                t_mask = tf.split(value=self.t_mask, **split_kwargs)
                t_mask = [tf.squeeze(weight) for weight in t_mask]

            loss = sequence_loss_tensor(self.out_tensor, self.ts, self.t_mask,
                    self.alphabet_size)

            with tf.variable_scope('regularization'):
                regularize = tf.contrib.layers.l2_regularizer(self.reg_scale)
                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                reg_term = sum([regularize(param) for param in params])

            loss += reg_term
            # Create TensorBoard scalar summary for loss
            tf.scalar_summary('train/loss', loss)

        self.loss = loss
