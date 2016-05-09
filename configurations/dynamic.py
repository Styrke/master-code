import sys
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import tensor_array_ops
from utils.tfextensions import sequence_loss_tensor
sys.path.insert(0, '..')
import model
import text_loader as tl


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

    def setup_placeholders(self):
        shape = [None, None]
        self.Xs       = tf.placeholder(tf.int32, shape=shape, name='X_input')
        self.ts       = tf.placeholder(tf.int32, shape=shape, name='t_input')
        self.ts_go    = tf.placeholder(tf.int32, shape=shape, name='t_input_go')
        self.X_len    = tf.placeholder(tf.int32, shape=[None], name='X_len')
        self.t_len    = tf.placeholder(tf.int32, shape=[None], name='t_len')
        self.feedback = tf.placeholder(tf.bool, name='feedback_indicator')
        self.t_mask   = tf.placeholder(tf.float32, shape=shape, name='t_mask')

        shape = [None, None]
        self.X_spaces     = tf.placeholder(tf.int32, shape=shape,  name='X_spaces')
        self.X_spaces_len = tf.placeholder(tf.int32, shape=[None], name='X_spaces_len')

    def build(self):
        print('Building model')
        self.embeddings = tf.Variable(
            tf.random_normal([self.alphabet_size, self.embedd_dims],
            stddev=0.1), name='embeddings')

        X_embedded = tf.gather(self.embeddings, self.Xs, name='embed_X')
        t_embedded = tf.gather(self.embeddings, self.ts_go, name='embed_t')

        with tf.variable_scope('dense_out'):
            W_out = tf.get_variable('W_out', [self.rnn_units, self.alphabet_size])
            b_out = tf.get_variable('b_out', [self.alphabet_size])

        with tf.variable_scope('encoder'):
            weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
            W_z = tf.get_variable('W_z',
                                  shape=[self.embedd_dims+self.rnn_units, self.rnn_units],
                                  initializer=weight_initializer)
            W_r = tf.get_variable('W_r',
                                  shape=[self.embedd_dims+self.rnn_units, self.rnn_units],
                                  initializer=weight_initializer)
            W_h = tf.get_variable('W_h',
                                  shape=[self.embedd_dims+self.rnn_units, self.rnn_units],
                                  initializer=weight_initializer)
            b_z = tf.get_variable('b_z',
                                  shape=[self.rnn_units],
                                  initializer=tf.constant_initializer())
            b_r = tf.get_variable('b_r',
                                  shape=[self.rnn_units],
                                  initializer=tf.constant_initializer())
            b_h = tf.get_variable('b_h',
                                  shape=[self.rnn_units],
                                  initializer=tf.constant_initializer())

            max_sequence_length = tf.reduce_max(self.X_len)
            min_sequence_length = tf.reduce_min(self.X_len)

            time = tf.constant(0)

            state_shape = tf.concat(0, [tf.expand_dims(tf.shape(self.X_len)[0], 0),
                                        tf.expand_dims(tf.constant(self.rnn_units), 0)])
            # state_shape = tf.Print(state_shape, [state_shape])
            state = tf.zeros(state_shape, dtype=tf.float32)

            inputs = tf.transpose(X_embedded, perm=[1, 0, 2])
            input_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)
            input_ta = input_ta.unpack(inputs)

            output_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)

            def encoder_cond(time, state, output_ta_t):
                return tf.less(time, max_sequence_length)

            def encoder_body(time, old_state, output_ta_t):
                x_t = input_ta.read(time)

                con = tf.concat(1, [x_t, old_state])
                z = tf.sigmoid(tf.matmul(con, W_z) + b_z)
                r = tf.sigmoid(tf.matmul(con, W_r) + b_r)
                con = tf.concat(1, [x_t, r*old_state])
                h = tf.tanh(tf.matmul(con, W_h) + b_h)
                new_state = (1-z)*h + z*old_state

                output_ta_t = output_ta_t.write(time, new_state)

                def updateall():
                    return new_state

                def updatesome():
                    return tf.select(tf.less(time, self.X_len), new_state, old_state)

                # TODO: only update state if seq_len < time
                state = tf.cond(tf.less(time, min_sequence_length), updateall, updatesome)

                return (time + 1, state, output_ta_t)

            loop_vars = [time, state, output_ta]

            time, state, output_ta = tf.while_loop(encoder_cond, encoder_body, loop_vars)

            enc_state = state

        tf.histogram_summary('final_encoder_state', enc_state)

        with tf.variable_scope('decoder'):
            weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
            W_z = tf.get_variable('W_z',
                                  shape=[self.embedd_dims+self.rnn_units, self.rnn_units],
                                  initializer=weight_initializer)
            W_r = tf.get_variable('W_r',
                                  shape=[self.embedd_dims+self.rnn_units, self.rnn_units],
                                  initializer=weight_initializer)
            W_h = tf.get_variable('W_h',
                                  shape=[self.embedd_dims+self.rnn_units, self.rnn_units],
                                  initializer=weight_initializer)
            b_z = tf.get_variable('b_z',
                                  shape=[self.rnn_units],
                                  initializer=tf.constant_initializer())
            b_r = tf.get_variable('b_r',
                                  shape=[self.rnn_units],
                                  initializer=tf.constant_initializer())
            b_h = tf.get_variable('b_h',
                                  shape=[self.rnn_units],
                                  initializer=tf.constant_initializer())

            max_sequence_length = tf.reduce_max(self.t_len)

            time = tf.constant(0)

            state_shape = tf.concat(0, [tf.expand_dims(tf.shape(self.X_len)[0], 0),
                                        tf.expand_dims(tf.constant(self.rnn_units), 0)])
            # state_shape = tf.Print(state_shape, [state_shape])
            state = tf.zeros(state_shape, dtype=tf.float32)

            inputs = tf.transpose(t_embedded, perm=[1, 0, 2])
            input_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)
            input_ta = input_ta.unpack(inputs)

            output_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)

            def decoder_cond(time, state, output_ta_t):
                return tf.less(time, max_sequence_length)

            def decoder_body_builder(feedback=False):
                def decoder_body(time, old_state, output_ta_t):
                    if feedback:
                        prev_1 = tf.matmul(state, W_out) + b_out
                        x_t = tf.gather(self.embeddings, tf.argmax(prev_1, 1))
                    else:
                        x_t = input_ta.read(time)

                    con = tf.concat(1, [x_t, old_state])
                    z = tf.sigmoid(tf.matmul(con, W_z) + b_z)
                    r = tf.sigmoid(tf.matmul(con, W_r) + b_r)
                    con = tf.concat(1, [x_t, r*old_state])
                    h = tf.tanh(tf.matmul(con, W_h) + b_h)
                    new_state = (1-z)*h + z*old_state

                    output_ta_t = output_ta_t.write(time, new_state)

                    return (time + 1, new_state, output_ta_t)
                return decoder_body

            loop_vars = [time, state, output_ta]

            _, state, output_ta = tf.while_loop(decoder_cond,
                                                decoder_body_builder(),
                                                loop_vars)
            _, valid_state, valid_output_ta = tf.while_loop(decoder_cond,
                                                            decoder_body_builder(feedback=True),
                                                            loop_vars)

            dec_state = state
            dec_out = tf.transpose(output_ta.pack(), perm=[1, 0, 2])
            valid_dec_out = tf.transpose(valid_output_ta.pack(), perm=[1, 0, 2])

        out_tensor = tf.reshape(dec_out, [-1, self.rnn_units])
        out_tensor = tf.matmul(out_tensor, W_out) + b_out
        out_shape = tf.concat(0, [tf.expand_dims(tf.shape(self.X_len)[0], 0),
                                  tf.expand_dims(max_sequence_length, 0),
                                  tf.expand_dims(tf.constant(self.alphabet_size), 0)])
        self.out_tensor = tf.reshape(out_tensor, out_shape)
        self.out_tensor.set_shape([None, None, self.alphabet_size])

        valid_out_tensor = tf.reshape(valid_dec_out, [-1, self.rnn_units])
        valid_out_tensor = tf.matmul(valid_out_tensor, W_out) + b_out
        # valid_out_shape = tf.concat(0, [tf.expand_dims(tf.shape(self.X_len)[0], 0),
        #                                 tf.expand_dims(max_sequence_length, 0),
        #                                 tf.expand_dims(tf.constant(self.alphabet_size), 0)])
        self.valid_out_tensor = tf.reshape(valid_out_tensor, out_shape)

        self.out = None

        # trans = tf.transpose(self.out_tensor, perm=[1, 0, 2])
        # self.out = tf.unpack(trans)

        # add TensorBoard summaries for all variables
        tf.contrib.layers.summarize_variables()

    def build_loss(self, out, out_tensor):
        """Build a loss function and accuracy for the model."""
        print('  Building loss and accuracy')

        with tf.variable_scope('accuracy'):
            argmax = tf.to_int32(tf.argmax(out_tensor, 2))
            correct = tf.to_float(tf.equal(argmax, self.ts)) * self.t_mask
            accuracy = tf.reduce_sum(correct) / tf.reduce_sum(self.t_mask)

        with tf.variable_scope('loss'):
            loss = sequence_loss_tensor(out_tensor, self.ts, self.t_mask,
                    self.alphabet_size)

            with tf.variable_scope('regularization'):
                regularize = tf.contrib.layers.l2_regularizer(self.reg_scale)
                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                reg_term = sum([regularize(param) for param in params])

            loss += reg_term

        return loss, accuracy

    def build_valid_loss(self):
        return self.build_loss(self.out, self.valid_out_tensor)

    def build_valid_prediction(self):
        return self.build_prediction(self.valid_out_tensor)

    def setup_batch_generators(self):
        """Load the datasets"""
        self.batch_generator = dict()

        # load training set
        print('Load training set')
        train_loader = tl.TextLoader(paths_X=self.train_x_files,
                                     paths_t=self.train_t_files,
                                     seq_len=self.seq_len)
        self.batch_generator['train'] = tl.TextBatchGenerator(
            loader=train_loader,
            batch_size=self.batch_size,
            use_dynamic_array_sizes=True,
            **self.schedule_kwargs)

        # load validation set
        print('Load validation set')
        valid_loader = tl.TextLoader(paths_X=self.valid_x_files,
                                     paths_t=self.valid_t_files,
                                     seq_len=self.seq_len)
        self.batch_generator['valid'] = tl.TextBatchGenerator(
            loader=valid_loader,
            batch_size=self.batch_size,
            use_dynamic_array_sizes=True)
