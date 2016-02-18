import tensorflow as tf
import numpy as np
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops import control_flow_ops

BATCH_SIZE = 2
EMBEDDING_SIZE = 3
MAX_SEQ_LENGTH = 5


def inference(alphabet_size, input, lengths):
    # character embeddings
    embeddings = tf.Variable(tf.random_uniform([alphabet_size,
                                                EMBEDDING_SIZE]))
    embedded = tf.gather(embeddings, input)

    # Pad the input so we are sure that we have something of the right length
    pad_size = MAX_SEQ_LENGTH - tf.shape(embedded)[1]
    pad_size = tf.reshape(pad_size, [1, 1])
    paddings = tf.pad(pad_size, np.array([[1, 1], [1, 0]]))
    rnn_pad = tf.pad(embedded, paddings)

    # Split the input into a list of tensors (one element per timestep)
    inputs = tf.split(1, MAX_SEQ_LENGTH, rnn_pad)
    inputs = [tf.squeeze(input_) for input_ in inputs]

    outputs = list()  # prepare to receive output

    for i in xrange(MAX_SEQ_LENGTH):
        zeros = tf.zeros([2, EMBEDDING_SIZE], dtype=tf.float32)
        output = tf.select(tf.less(i, lengths), inputs[i], zeros)
        outputs.append(output)

    outputs = tf.transpose(tf.pack(outputs), perm=[1, 0, 2])
    indices = tf.reshape(lengths-1, [2, 1])
    output = _grid_gather(outputs, indices)

    return output


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
    ones_shape = tf.concat(0, [tf.expand_dims(tf.constant(1), 0),
                               tf.expand_dims(indices_shape[1], 0)])
    ones = tf.ones(ones_shape, dtype=tf.int32)
    rng_array = tf.matmul(rng, ones)
    indices = indices + rng_array

    # gather and return
    return tf.gather(flat_params, indices)
