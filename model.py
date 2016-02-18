import tensorflow as tf
import numpy as np
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops import control_flow_ops

EMBEDDING_SIZE = 3
MAX_SEQ_LENGTH = 5

RNN_HID_UNITS = 10
RNN_OUT_UNITS = 10

def inference(alphabet_size, input, lengths):
    batch_size = tf.expand_dims(tf.shape(input)[0], 0, name='get_batch_size')

    # character embeddings
    embeddings = tf.Variable(tf.random_uniform([alphabet_size,
                                                EMBEDDING_SIZE]),
                             name='embeddings')
    embedded = tf.gather(embeddings, input)

    # Pad the input so we are sure that we have something of the right length
    pad_size = MAX_SEQ_LENGTH - tf.shape(embedded)[1]
    pad_size = tf.reshape(pad_size, [1, 1], name='pad_size')
    paddings = tf.pad(pad_size, np.array([[1, 1], [1, 0]]), name='paddings')
    rnn_pad = tf.pad(embedded, paddings, name='rnn_pad')

    # Split the input into a list of tensors (one element per timestep)
    inputs = tf.split(1, MAX_SEQ_LENGTH, rnn_pad, name='split_input_per_timestep')
    inputs = [tf.squeeze(input_) for input_ in inputs]

    zeros_shape = tf.concat(0, [batch_size, tf.constant(EMBEDDING_SIZE, shape=[1])])
    zeros = tf.zeros(zeros_shape, dtype=tf.float32, name='bias')

    W_ih = tf.Variable(tf.truncated_normal([EMBEDDING_SIZE, RNN_HID_UNITS]), name='W_ih')
    W_hh = tf.Variable(tf.truncated_normal([RNN_HID_UNITS, RNN_HID_UNITS]), name='W_hh')
    W_ho = tf.Variable(tf.truncated_normal([RNN_HID_UNITS, RNN_OUT_UNITS]), name='W_ho')
    b_h = tf.Variable(tf.zeros([RNN_HID_UNITS]), name='b_h')
    b_o = tf.Variable(tf.zeros([RNN_OUT_UNITS]), name='b_o')

    outputs = list()  # prepare to receive output
    state_shape = tf.concat(0, [batch_size, tf.constant(RNN_HID_UNITS, shape=[1])], name='state_shape')
    state = tf.zeros(state_shape, name='state')

    for i in xrange(MAX_SEQ_LENGTH):
        state = tf.tanh(tf.matmul(inputs[i], W_ih, name='input_Whi') + tf.matmul(state, W_hh, name='state_Whh') + b_h, name='state_calc')
        output = tf.matmul(state, W_ho, name='output_mult') + b_o
        outputs.append(output)

    outputs = tf.transpose(tf.pack(outputs), perm=[1, 0, 2], name='transpose_output')
    indices_shape = tf.concat(0, [batch_size, [1]], name='indices_shape')
    indices = tf.reshape(lengths-1, indices_shape, name='indices_reshape')
    output = _grid_gather(outputs, indices)

    return output


def loss(logits, targets):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits, targets)


def _grid_gather(params, indices):
    indices_shape = tf.shape(indices, name='indices_shape')
    params_shape = tf.shape(params, name='params_shape')

    # reshape params
    flat_params_dim0 = tf.reduce_prod(params_shape[:2], name='flat_params_dim0')
    flat_params_dim0_exp = tf.expand_dims(flat_params_dim0, 0, name='flat_params_dim0_exp')
    flat_params_shape = tf.concat(0, [flat_params_dim0_exp, params_shape[2:]], name='flat_params_shape')
    flat_params = tf.reshape(params, flat_params_shape, name='flat_params')

    # fix indices
    rng = tf.expand_dims(tf.range(flat_params_dim0, delta=params_shape[1]), 1, name='range')
    ones_shape = tf.concat(0, [tf.expand_dims(tf.constant(1), 0),
                               tf.expand_dims(indices_shape[1], 0)])
    ones = tf.ones(ones_shape, dtype=tf.int32)
    rng_array = tf.matmul(rng, ones, name='range_array')
    indices = indices + rng_array

    # gather and return
    return tf.gather(flat_params, indices, name='grid_gather_result')
