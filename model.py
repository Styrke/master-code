import tensorflow as tf
import numpy as np
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops import control_flow_ops

EMBEDDING_SIZE = 3
MAX_SEQ_LENGTH = 5

RNN_HID_UNITS = 10
RNN_OUT_UNITS = 10

def inference(alphabet_size, input, lengths):
    batch_size = tf.expand_dims(tf.shape(input)[0], 0)

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

    zeros_shape = tf.concat(0, [batch_size, tf.constant(EMBEDDING_SIZE, shape=[1])])
    zeros = tf.zeros(zeros_shape, dtype=tf.float32)

    W_ih = tf.Variable(tf.truncated_normal([EMBEDDING_SIZE, RNN_HID_UNITS]))
    W_hh = tf.Variable(tf.truncated_normal([RNN_HID_UNITS, RNN_HID_UNITS]))
    W_ho = tf.Variable(tf.truncated_normal([RNN_HID_UNITS, RNN_OUT_UNITS]))
    b_h = tf.Variable(tf.zeros([RNN_HID_UNITS]))
    b_o = tf.Variable(tf.zeros([RNN_OUT_UNITS]))

    outputs = list()  # prepare to receive output
    state_shape = tf.concat(0, [batch_size, tf.constant(RNN_HID_UNITS, shape=[1])])
    state = tf.zeros(state_shape)

    for i in xrange(MAX_SEQ_LENGTH):
        state = tf.tanh(tf.matmul(inputs[i], W_ih) + tf.matmul(state, W_hh) + b_h)
        output = tf.matmul(state, W_ho) + b_o
        outputs.append(output)

    outputs = tf.transpose(tf.pack(outputs), perm=[1, 0, 2])
    indices_shape = tf.concat(0, [batch_size, [1]])
    indices = tf.reshape(lengths-1, indices_shape)
    output = _grid_gather(outputs, indices)

    return output


def loss(logits, targets):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)


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
