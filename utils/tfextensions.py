import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops

# TODO: put in custom_ops file
# TODO: documentation ..!
def _grid_gather(params, indices):
    indices_shape = tf.shape(indices)
    params_shape = tf.shape(params)

    # reshape params
    flat_params_dim0 = tf.reduce_prod(params_shape[:2])
    flat_params_dim0_exp = tf.expand_dims(flat_params_dim0, 0)
    flat_params_shape = tf.concat(0, [flat_params_dim0_exp, params_shape[2:]])
    flat_params = tf.reshape(params, flat_params_shape)

    # fix indices
    rng = tf.expand_dims(tf.range(flat_params_dim0, delta=params_shape[1]), 1)
    ones_shape_list = [
        tf.expand_dims(tf.constant(1), 0),
        tf.expand_dims(indices_shape[1], 0)
    ]
    ones_shape = tf.concat(0, ones_shape_list)
    ones = tf.ones(ones_shape, dtype=tf.int32)
    rng_array = tf.matmul(rng, ones)
    indices = indices + rng_array

    # gather and return
    return tf.gather(flat_params, indices)


def sequence_loss_tensor(logits, targets, weights, num_classes,
                         average_across_timesteps=True,
                         softmax_loss_function=None, name=None):
    """Weighted cross-entropy loss for a sequence of logits (per example).

    """
#    if (logits.get_shape()[0:2]) != targets.get_shape() \
#        or (logits.get_shape()[0:2]) != weights.get_shape():
#        print(logits.get_shape()[0:2])
#        print(targets.get_shape())
#        print(weights.get_shape())
#        raise ValueError("Shapes of logits, weights, and targets must be the "
#            "same")
    with ops.op_scope([logits, targets, weights], name, "sequence_loss_by_example"):
        probs_flat = tf.reshape(logits, [-1, num_classes])
        targets = tf.reshape(targets, [-1])
        if softmax_loss_function is None:
            crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
                    probs_flat, targets)
        else:
            crossent = softmax_loss_function(probs_flat, targets)
        crossent = crossent * tf.reshape(weights, [-1])
        crossent = tf.reduce_sum(crossent)
        total_size = math_ops.reduce_sum(weights)
        total_size += 1e-12 # to avoid division by zero
        crossent /= total_size
        return crossent

def mask(sequence_lengths):
    # based on this SO answer: http://stackoverflow.com/a/34138336/118173
    batch_size = tf.shape(sequence_lengths)[0]
    max_len = tf.reduce_max(sequence_lengths)

    lengths_transposed = tf.expand_dims(sequence_lengths, 1)
    lengths_tiled = tf.tile(lengths_transposed, tf.pack([1, max_len]))

    rng = tf.range(max_len)
    rng_row = tf.expand_dims(rng, 0)
    rng_tiled = tf.tile(rng_row, tf.pack([batch_size, 1]))

    return tf.less(rng_tiled, max_len)
