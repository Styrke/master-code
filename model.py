import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn

class Model(object):

    def __init__(self, alphabet_size, embedd_dims=8, max_x_seq_len=25,
        max_t_seq_len=25, rnn_units=100):
        self.alphabet_size = alphabet_size
        self.embedd_dims = embedd_dims
        self.max_x_seq_len = max_x_seq_len
        self.max_t_seq_len = max_t_seq_len
        self.rnn_units = rnn_units

    def build(self, Xs, X_len, ts):
        print('Building model')

        self.embeddings = tf.Variable(
            tf.random_uniform([self.alphabet_size, self.embedd_dims]),
            name='embeddings')
        tf.histogram_summary('embeddings', self.embeddings)

        X_embedded = tf.gather(self.embeddings, Xs, name='embed_X')
        t_embedded = tf.gather(self.embeddings, ts, name='embed_t')

        with tf.variable_scope('split_X_inputs'):
            X_list = tf.split(
                split_dim=1,
                num_split=self.max_x_seq_len,
                value=X_embedded)

            X_list = [tf.squeeze(X) for X in X_list]

            [X.set_shape([None, self.embedd_dims]) for X in X_list]

        with tf.variable_scope('split_t_inputs'):
            t_list = tf.split(
                split_dim=1,
                num_split=self.max_t_seq_len,
                value=t_embedded)

            t_list = [tf.squeeze(t) for t in t_list]

            [t.set_shape([None, self.embedd_dims]) for t in t_list]

        with tf.variable_scope('dense_out'):
            W_out = tf.get_variable('W_out',
                [self.rnn_units, self.alphabet_size])
            b_out = tf.get_variable('b_out', [self.alphabet_size])

        cell = rnn_cell.GRUCell(self.rnn_units)

        # encoder
        enc_outputs, enc_state = rnn.rnn(
            cell=cell,
            inputs=X_list,
            dtype=tf.float32,
            sequence_length=X_len,
            scope='rnn_encoder')

        # decoder
        dec_out, dec_state = seq2seq.rnn_decoder(
            decoder_inputs=t_list,
            initial_state=enc_state,
            cell=cell)

        self.out = []
        for d in dec_out:
            self.out.append(tf.matmul(d, W_out) + b_out)

        # for debugging network
        out_packed = tf.pack(self.out)
        out_packed = tf.transpose(out_packed, perm=[1, 0, 2])
        print(out_packed.get_shape())
        self.out_tensor = out_packed



    def build_loss(self, ts, t_mask):
        print('Building model loss')
        # TODO: build some fail safes, e.g. check if model has built
        with tf.variable_scope('loss'):
            ts = tf.split(
                split_dim=1, num_split=self.max_t_seq_len, value=ts)

            with tf.variable_scope('split_t_mask'):
                t_mask = tf.split(
                    split_dim=1,
                    num_split=self.max_t_seq_len,
                    value=t_mask)

                t_mask = [tf.squeeze(weight) for weight in t_mask]

            loss = seq2seq.sequence_loss(
                self.out,
                ts,
                t_mask,
                self.max_t_seq_len)

        self.loss = loss


    def build_prediction(self):
        with tf.variable_scope('prediction'):
            # logits is a list of tensors of shape [batch_size,
            # alphabet_size]. We need a tensor shape [batch_size,
            # target_seq_len, alphabet_size]
            packed_logits = tf.transpose(tf.pack(self.out), perm=[1, 0, 2])

            self.ys = tf.argmax(packed_logits, dimension=2)


    def training(self, learning_rate):
        print('Building model training')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


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
