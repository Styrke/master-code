import sys
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import tensor_array_ops
sys.path.insert(0, '..')
import model


class Model(model.Model):
    """Configuration where the model has a 2-layer RNN encoder-decoder."""
    # overwrite config
    batch_size = 32
    seq_len = 25
    rnn_units = 100
    valid_freq = 100  # Using a smaller valid set for debug so can do it more frequently.

    # only use a single of the validation files for this debugging config
    valid_x_files = ['data/valid/devtest2006.en']
    valid_t_files = ['data/valid/devtest2006.da']

    def setup_placeholders(self):
        shape = [None, self.max_x_seq_len]
        self.Xs       = tf.placeholder(tf.int32, shape=shape, name='X_input')
        self.ts       = tf.placeholder(tf.int32, shape=shape, name='t_input')
        self.ts_go    = tf.placeholder(tf.int32, shape=shape, name='t_input_go')
        self.X_len    = tf.placeholder(tf.int32, shape=[None], name='X_len')
        self.t_len    = tf.placeholder(tf.int32, shape=[None], name='t_len')
        self.feedback = tf.placeholder(tf.bool, name='feedback_indicator')
        self.t_mask   = tf.placeholder(tf.float32, shape=shape, name='t_mask')

        shape = [None, self.max_x_seq_len//4]
        self.X_spaces     = tf.placeholder(tf.int32, shape=shape,  name='X_spaces')
        self.X_spaces_len = tf.placeholder(tf.int32, shape=[None], name='X_spaces_len')

    def build(self):
        print('  Building model')
        self.embeddings = tf.Variable(
            tf.random_normal([self.alphabet_size, self.embedd_dims],
            stddev=0.1), name='embeddings')

        X_embedded = tf.gather(self.embeddings, self.Xs,    name='embed_X')
        t_embedded = tf.gather(self.embeddings, self.ts_go, name='embed_t')

        with tf.variable_scope('dense_out'):
            W_out = tf.get_variable('W_out', [self.rnn_units, self.alphabet_size])
            b_out = tf.get_variable('b_out', [self.alphabet_size])

        cell = rnn_cell.GRUCell(self.rnn_units)

        # encoder
        enc_outputs, enc_state = (
                rnn.dynamic_rnn(cell=cell,
                    inputs=X_embedded,
                    dtype=tf.float32,
                    sequence_length=self.X_len,
                    scope='rnn_encoder') )

        tf.histogram_summary('final_encoder_state', enc_state)

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
        dec_out, dec_state = (
                rnn.dynamic_rnn(cell=cell,
                    inputs=t_embedded,
                    initial_state=enc_state,
                    scope='rnn_decoder') )
                    #loop_function=decoder_loop_function) )

        with tf.variable_scope('split_dec_out'):
            dec_list = tf.split(split_dim=1,
                                num_split=self.max_t_seq_len,
                                value=dec_out)
            dec_list = [tf.squeeze(d) for d in dec_list]
            [d.set_shape([None, self.rnn_units]) for d in dec_list]

        self.out = [tf.matmul(d, W_out) + b_out for d in dec_list]

        # for debugging network (NOTE should write this outside of build)
        out_packed = tf.pack(self.out)
        out_packed = tf.transpose(out_packed, perm=[1, 0, 2])
        self.out_tensor = out_packed

        # add TensorBoard summaries for all variables
        tf.contrib.layers.summarize_variables()


    def valid_dict(self, batch, feedback=None):
        """ Return feed_dict for validation """
        return { self.Xs:     batch['x_encoded'],
                 self.ts:     batch['t_encoded'],
                 self.ts_go:  batch['t_encoded_go'],
                 self.X_len:  batch['x_len'],
                 self.t_len:  batch['t_len'],
                 self.t_mask: batch['t_mask'],
                 self.feedback: feedback or True,
                 self.X_spaces: batch['x_spaces'],
                 self.X_spaces_len: batch['x_spaces_len'] }
