import sys
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import tensor_array_ops
sys.path.insert(0, '..')
import model


class Model(model.Model):
    """Configuration where the model has a 2-layer RNN encoer."""
    # overwrite config
    batch_size = 32
    seq_len = 25
    rnn_units = 100
    valid_freq = 100  # Using a smaller valid set for debug so can do it more frequently.

    # only use a single of the validation files for this debugging config
    valid_x_files = ['data/valid/devtest2006.en']
    valid_t_files = ['data/valid/devtest2006.da']

    def build(self):
        print('Building model')
        self.embeddings = tf.Variable(
            tf.random_normal([self.alphabet_size, self.embedd_dims],
            stddev=0.1), name='embeddings')

        X_embedded = tf.gather(self.embeddings, self.Xs, name='embed_X')
        t_embedded = tf.gather(self.embeddings, self.ts_go, name='embed_t')

        with tf.variable_scope('split_t_inputs'):
            t_list = tf.split(
                split_dim=1,
                num_split=self.max_t_seq_len,
                value=t_embedded)

            t_list = [tf.squeeze(t) for t in t_list]

            [t.set_shape([None, self.embedd_dims]) for t in t_list]

        with tf.variable_scope('dense_out'):
            W_out = tf.get_variable('W_out', [self.rnn_units, self.alphabet_size])
            b_out = tf.get_variable('b_out', [self.alphabet_size])

        with tf.variable_scope('encoder'):
            weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
            W_h = tf.get_variable('W_h',
                                  shape=[self.rnn_units, self.rnn_units],
                                  initializer=weight_initializer)
            W_x = tf.get_variable('W_x',
                                  shape=[self.embedd_dims, self.rnn_units],
                                  initializer=weight_initializer)
            b = tf.get_variable('b',
                                shape=[self.rnn_units],
                                initializer=tf.constant_initializer())

            # TODO: use this instead of self.max_x_seq_len
            max_sequence_length = tf.reduce_max(self.X_len)

            time = tf.constant(0)

            state_shape = tf.concat(0, [tf.expand_dims(tf.shape(self.X_len)[0], 0),
                                        tf.expand_dims(tf.constant(self.rnn_units), 0)])
            # state_shape = tf.Print(state_shape, [state_shape])
            state = tf.zeros(state_shape, dtype=tf.float32)

            inputs = tf.transpose(X_embedded, perm=[1, 0, 2])
            input_ta = tensor_array_ops.TensorArray(tf.float32, self.max_x_seq_len)
            input_ta = input_ta.unpack(inputs)

            output_ta = tensor_array_ops.TensorArray(tf.float32, self.max_x_seq_len)

            def encoder_cond(time, state, output_ta_t):
                return tf.less(time, self.max_x_seq_len)

            def encoder_body(time, state, output_ta_t):
                x_t = input_ta.read(time)

                state = tf.tanh(tf.matmul(state, W_h) + tf.matmul(x_t, W_x) + b)

                output_ta_t = output_ta_t.write(time, state)

                # TODO: only update state if seq_len < time

                return (time + 1, state, output_ta_t)

            loop_vars = [time, state, output_ta]

            time, state, output_ta = tf.while_loop(encoder_cond, encoder_body, loop_vars)

            enc_state = state

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
        cell = rnn_cell.GRUCell(self.rnn_units)
        dec_out, dec_state = seq2seq.rnn_decoder(
            decoder_inputs=t_list,
            initial_state=enc_state,
            cell=cell,
            loop_function=decoder_loop_function)

        self.out = []
        for d in dec_out:
            self.out.append(tf.matmul(d, W_out) + b_out)

        # for debugging network (should write this outside of build)
        out_packed = tf.pack(self.out)
        out_packed = tf.transpose(out_packed, perm=[1, 0, 2])
        self.out_tensor = out_packed

        # add TensorBoard summaries for all variables
        tf.contrib.layers.summarize_variables()
