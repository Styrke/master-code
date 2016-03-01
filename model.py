import tensorflow as tf
import numpy as np
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn

EMBEDD_DIMS = 3  # number of dimensions for each char embedding

MAX_X_SEQ_LEN = 25  # max length of x input sequence
MAX_T_SEQ_LEN = 25  # max length of t truth sequence

RNN_UNITS = 201  # this should currently be equal to the output alphabet size


def inference(alphabet_size, input, input_lengths, target):
    print 'Building model inference'

    # character embeddings
    embeddings = tf.Variable(tf.random_uniform([alphabet_size, EMBEDD_DIMS]))

    # encoder_inputs
    x_embedded = tf.gather(embeddings, input)
    encoder_inputs = tf.split(
        split_dim=1,
        num_split=MAX_X_SEQ_LEN,
        value=x_embedded)
    encoder_inputs = [tf.squeeze(x) for x in encoder_inputs]
    [x.set_shape([None, EMBEDD_DIMS]) for x in encoder_inputs]

    # decoder_inputs
    t_embedded = tf.gather(embeddings, target)
    decoder_inputs = tf.split(
        split_dim=1,
        num_split=MAX_T_SEQ_LEN,
        value=t_embedded)
    decoder_inputs = [tf.squeeze(x) for x in decoder_inputs]
    [x.set_shape([None, EMBEDD_DIMS]) for x in decoder_inputs]

    cell = rnn_cell.BasicRNNCell(RNN_UNITS)

    # encoder
    enc_outputs, enc_state = rnn.rnn(
        cell=cell,
        inputs=encoder_inputs,
        dtype=tf.float32,
        sequence_length=input_lengths)

    # decoder
    dec_outputs, dec_state = seq2seq.rnn_decoder(
        decoder_inputs=decoder_inputs,
        initial_state=enc_state,
        cell=cell)

    outputs = dec_outputs

    return outputs


def loss(logits, target, target_mask):
    print 'Building model loss'

    targets = tf.split(split_dim=1, num_split=MAX_T_SEQ_LEN, value=target)

    weights = tf.split(
        split_dim=1,
        num_split=MAX_T_SEQ_LEN,
        value=target_mask)

    weights = [tf.squeeze(weight) for weight in weights]

    loss = seq2seq.sequence_loss(
        logits,
        targets,
        weights,
        MAX_T_SEQ_LEN)

    return loss


def prediction(logits):
    # logits is a list of tensors of shape [batch_size, alphabet_size]. We need
    # a single tensor shape [batch_size, target_seq_len, alphabet_size]
    packed_logits = tf.transpose(tf.pack(logits), perm=[1, 0, 2])

    return tf.argmax(packed_logits, dimension=2)


def training(loss, learning_rate):
    print 'Building model training'

    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


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
