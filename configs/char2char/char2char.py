import tensorflow as tf

from configs import default
from utils.rnn import encoder
from utils.rnn import attention_decoder


class Model(default.Model):

    def build(self):
        print('Building model')
        self.x_embeddings = tf.Variable(
            tf.random_normal([self.alphabet_size, self.embedd_dims],
            stddev=0.1), name='x_embeddings')
        self.t_embeddings = tf.Variable(
            tf.random_normal([self.alphabet_size, self.embedd_dims],
            stddev=0.1), name='t_embeddings')

        X_embedded = tf.gather(self.x_embeddings, self.Xs, name='embed_X')
        t_embedded = tf.gather(self.t_embeddings, self.ts_go, name='embed_t')

        with tf.variable_scope('dense_out'):
            W_out = tf.get_variable('W_out', [self.word_encoder_units, self.alphabet_size])
            b_out = tf.get_variable('b_out', [self.alphabet_size])

        # forward encoding
        char_enc_state, char_enc_out = encoder(X_embedded, self.X_len, 'char_encoder', self.char_encoder_units)

        # decoding
        dec_state, dec_out, valid_dec_out = (
            attention_decoder(char_enc_out, self.X_len, char_enc_state,
                              t_embedded, self.t_len, self.word_encoder_units,
                              self.t_embeddings, W_out, b_out))

        out_tensor = tf.reshape(dec_out, [-1, self.word_encoder_units])
        out_tensor = tf.matmul(out_tensor, W_out) + b_out
        out_shape = tf.concat(0, [tf.expand_dims(tf.shape(self.X_len)[0], 0),
                                  tf.expand_dims(tf.shape(t_embedded)[1], 0),
                                  tf.expand_dims(tf.constant(self.alphabet_size), 0)])
        self.out_tensor = tf.reshape(out_tensor, out_shape)
        self.out_tensor.set_shape([None, None, self.alphabet_size])

        valid_out_tensor = tf.reshape(valid_dec_out, [-1, self.word_encoder_units])
        valid_out_tensor = tf.matmul(valid_out_tensor, W_out) + b_out
        self.valid_out_tensor = tf.reshape(valid_out_tensor, out_shape)

        self.out = None

        # add TensorBoard summaries for all variables
        tf.contrib.layers.summarize_variables()
