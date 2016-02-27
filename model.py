import tensorflow as tf
import numpy as np
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn

EMBEDDING_SIZE = 3

MAX_IN_SEQ_LENGTH = 5
MAX_OUT_SEQ_LENGTH = 5

RNN_UNITS = 15

def inference(alphabet_size, input, target):
    # character embeddings
    embeddings = tf.Variable(
            tf.random_uniform(
                [alphabet_size, EMBEDDING_SIZE]),
            name='embeddings')

    # encoder_inputs
    x_embedded = tf.gather(embeddings, input)
    encoder_inputs = tf.split(
            split_dim=1,
            num_split=MAX_IN_SEQ_LENGTH,
            value=x_embedded,
            name='encoder_embeddings')
    encoder_inputs = [tf.squeeze(x) for x in encoder_inputs]

    # decoder_inputs
    t_embedded = tf.gather(embeddings, target)
    decoder_inputs = tf.split(
            split_dim=1,
            num_split=MAX_OUT_SEQ_LENGTH,
            value=t_embedded,
            name='decoder_embeddings')
    decoder_inputs = [tf.squeeze(x) for x in decoder_inputs]

    cell = rnn_cell.BasicRNNCell(RNN_UNITS)

    # encoder
    enc_outputs, enc_state = rnn.rnn(cell, encoder_inputs, dtype=tf.float32)

    # decoder
    dec_outputs, dec_state = seq2seq.rnn_decoder(decoder_inputs, enc_state, cell)

    outputs = dec_outputs

    targets = tf.split(
            split_dim=1,
            num_split=MAX_OUT_SEQ_LENGTH,
            value=target,
            name='truth')
    weights = tf.ones_like(
            targets[0],
            dtype=tf.float32,
            name='loss_weights')
    weights = [weights] * MAX_OUT_SEQ_LENGTH

    loss = seq2seq.sequence_loss(
                outputs,
                targets,
                weights,
                MAX_OUT_SEQ_LENGTH)

    return outputs, loss

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
