import sys
sys.path.insert(0, '..')
import model
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn

# hyperparams
rnn_units = 400
embedd_dims = 16
l1_units = 400


class ConfigModel(model.Model):

    def build(self):
        print('Building model')
        self.embeddings = tf.Variable(
            tf.random_uniform([self.alphabet_size, embedd_dims]),
            name='embeddings')

        X_embedded = tf.gather(self.embeddings, self.Xs, name='embed_X')
        t_embedded = tf.gather(self.embeddings, self.ts_go, name='embed_t')

        with tf.variable_scope('split_X_inputs'):
            X_list = tf.split(
                split_dim=1,
                num_split=self.max_x_seq_len,
                value=X_embedded)

            X_list = [tf.squeeze(X) for X in X_list]

            [X.set_shape([None, embedd_dims]) for X in X_list]

        with tf.variable_scope('split_t_inputs'):
            t_list = tf.split(
                split_dim=1,
                num_split=self.max_t_seq_len,
                value=t_embedded)

            t_list = [tf.squeeze(t) for t in t_list]

            [t.set_shape([None, embedd_dims]) for t in t_list]

        with tf.variable_score('dense_l1'):
            W_l1 = tf.get_variable('W_l1',
                [rnn_units, l1_units])
            b_l1 = tf.get_variable('b_l1', [l1_units])

        with tf.variable_scope('dense_out'):
            W_out = tf.get_variable('W_out',
                [l1_units, self.alphabet_size])
            b_out = tf.get_variable('b_out', [self.alphabet_size])

        cell = rnn_cell.GRUCell(rnn_units)

        # encoder
        enc_outputs, enc_state = rnn.rnn(
            cell=cell,
            inputs=X_list,
            dtype=tf.float32,
            sequence_length=self.X_len,
            scope='rnn_encoder')

        # The loop function provides inputs to the decoder:
        def decoder_loop_function(prev, i):
            def feedback_on():
                prev_1 = tf.matmul(prev, W_out) + b_out
                # feedback is on, so feed the decoder with the previous output
                return tf.gather(self.embeddings, tf.argmax(prev_1, 1))

            def feedback_off():
                # feedback is off, so just feed the decoder with t's
                return t_list[i]

            return tf.cond(self.feedback, feedback_on, feedback_off)

        # decoder
        dec_out, dec_state = seq2seq.rnn_decoder(
            decoder_inputs=t_list,
            initial_state=enc_state,
            cell=cell,
            loop_function=decoder_loop_function)

        l1 = []
        for t in dec_out:
            self.l1.append(tf.matmul(t, W_l1) + b_l1)

        self.out = []
        for t in l1:
            self.out.append(tf.matmul(t, W_out) + b_out)

        # for debugging network (should write this outside of build)
        out_packed = tf.pack(self.out)
        out_packed = tf.transpose(out_packed, perm=[1, 0, 2])
        print(out_packed.get_shape())
        self.out_tensor = out_packed

        # add TensorBoard summaries for all variables
        tf.contrib.layers.summarize_variables()
