import tensorflow as tf
import text_loader as tl
from utils.tfextensions import sequence_loss_tensor
from utils.tfextensions import _grid_gather
from utils.tfextensions import mask
from utils.rnn import encoder
from utils.rnn import attention_decoder
from data.alphabet import Alphabet

from configs import default


class Model(default.Model):
    attn_units = 400

    def build(self):
        print('Building model')
        self.x_embeddings = tf.Variable(
            tf.random_normal([self.alphabet_src_size, self.embedd_dims],
            stddev=0.1), name='x_embeddings')
        self.t_embeddings = tf.Variable(
            tf.random_normal([self.alphabet_tar_size, self.embedd_dims],
            stddev=0.1), name='t_embeddings')

        X_embedded = tf.gather(self.x_embeddings, self.Xs, name='embed_X')
        t_embedded = tf.gather(self.t_embeddings, self.ts_go, name='embed_t')

        with tf.variable_scope('dense_out'):
            W_out = tf.get_variable('W_out', [self.word_encoder_units*2, self.alphabet_tar_size])
            b_out = tf.get_variable('b_out', [self.alphabet_tar_size])

        # forward encoding
        char_enc_state, char_enc_out = encoder(X_embedded, self.X_len, 'char_encoder', self.char_encoder_units)
        char2word = _grid_gather(char_enc_out, self.X_spaces)
        char2word.set_shape([None, None, self.char_encoder_units])
        word_enc_state, word_enc_out = encoder(char2word, self.X_spaces_len, 'word_encoder', self.word_encoder_units)

        # backward encoding words
        char2word = tf.reverse_sequence(char2word, tf.to_int64(self.X_spaces_len), 1)
        char2word.set_shape([None, None, self.char_encoder_units])
        word_enc_state_bck, word_enc_out_bck = encoder(char2word, self.X_spaces_len, 'word_encoder_backwards', self.word_encoder_units)
        word_enc_out_bck = tf.reverse_sequence(word_enc_out_bck, tf.to_int64(self.X_spaces_len), 1)

        word_enc_state = tf.concat(1, [word_enc_state, word_enc_state_bck])
        word_enc_out = tf.concat(2, [word_enc_out, word_enc_out_bck])

        # decoding
        dec_state, dec_out, valid_dec_out = (
            attention_decoder(word_enc_out, self.X_spaces_len, word_enc_state,
                              t_embedded, self.t_len, self.attn_units,
                              self.t_embeddings, W_out, b_out))

        out_tensor = tf.reshape(dec_out, [-1, self.word_encoder_units*2])
        out_tensor = tf.matmul(out_tensor, W_out) + b_out
        out_shape = tf.concat(0, [tf.expand_dims(tf.shape(self.X_len)[0], 0),
                                  tf.expand_dims(tf.shape(t_embedded)[1], 0),
                                  tf.expand_dims(tf.constant(self.alphabet_tar_size), 0)])
        self.out_tensor = tf.reshape(out_tensor, out_shape)
        self.out_tensor.set_shape([None, None, self.alphabet_tar_size])

        valid_out_tensor = tf.reshape(valid_dec_out, [-1, self.word_encoder_units*2])
        valid_out_tensor = tf.matmul(valid_out_tensor, W_out) + b_out
        self.valid_out_tensor = tf.reshape(valid_out_tensor, out_shape)

        self.out = None

        # add TensorBoard summaries for all variables
        tf.contrib.layers.summarize_variables()
