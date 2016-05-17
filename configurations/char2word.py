import sys
import tensorflow as tf
sys.path.insert(0, '../..')
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn

import model
from utils.tfextensions import _grid_gather


class Model(model.Model):
    """Configuration where the model has a char2word encoder."""
    char_enc_units = 100
    word_enc_units = 100
    dec_units = 100

    def build(self):
        print('Building model')
        self.embeddings = tf.Variable(
            tf.random_uniform([self.alphabet_size, self.embedd_dims]),
            name='embeddings')

        X_embedded = tf.gather(self.embeddings, self.Xs, name='embed_X')
        t_embedded = tf.gather(self.embeddings, self.ts_go, name='embed_t')

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
            W_out = tf.get_variable('W_out', [self.dec_units, self.alphabet_size])
            b_out = tf.get_variable('b_out', [self.alphabet_size])

        # char encoder
        char_cell = rnn_cell.GRUCell(self.char_enc_units)
        char_enc_outputs, char_enc_state = rnn.rnn(
            cell=char_cell,
            inputs=X_list,
            dtype=tf.float32,
            sequence_length=self.X_len,
            scope='rnn_char_encoder')

        # char2word
        char2word = tf.transpose(tf.pack(char_enc_outputs), perm=[1, 0, 2])
        char2word = _grid_gather(char2word, self.X_spaces)
        char2word = tf.unpack(tf.transpose(char2word, perm=[1, 0, 2]))

        [t.set_shape([None, self.char_enc_units]) for t in char2word]

        # word encoder
        word_cell = rnn_cell.GRUCell(self.word_enc_units)
        word_enc_outputs, word_enc_state = rnn.rnn(
            cell=word_cell,
            inputs=char2word,
            dtype=tf.float32,
            sequence_length=self.X_spaces_len,
            scope='rnn_word_encoder'
        )

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
        att_states = tf.transpose(tf.pack(word_enc_outputs), perm=[1, 0, 2])
        dec_cell = rnn_cell.GRUCell(self.dec_units)
        dec_out, dec_state = seq2seq.attention_decoder(
            decoder_inputs=t_list,
            initial_state=word_enc_state,
            attention_states=att_states,
            cell=dec_cell,
            loop_function=decoder_loop_function,
            scope='attention_decoder'
        )

        self.out = []
        for d in dec_out:
            self.out.append(tf.matmul(d, W_out) + b_out)

        # for debugging network (should write this outside of build)
        out_packed = tf.pack(self.out)
        out_packed = tf.transpose(out_packed, perm=[1, 0, 2])
        self.out_tensor = out_packed

        # add TensorBoard summaries for all variables
        tf.contrib.layers.summarize_variables()
