import sys
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from tensorflow.python.ops import tensor_array_ops
from utils.tfextensions import sequence_loss_tensor
from utils.tfextensions import _grid_gather
from utils.tfextensions import mask
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

    def setup_placeholders(self):
        shape = [None, None]
        self.Xs       = tf.placeholder(tf.int32, shape=shape, name='X_input')
        self.ts       = tf.placeholder(tf.int32, shape=shape, name='t_input')
        self.ts_go    = tf.placeholder(tf.int32, shape=shape, name='t_input_go')
        self.X_len    = tf.placeholder(tf.int32, shape=[None], name='X_len')
        self.t_len    = tf.placeholder(tf.int32, shape=[None], name='t_len')
        self.feedback = tf.placeholder(tf.bool, name='feedback_indicator')
        self.x_mask   = tf.placeholder(tf.float32, shape=shape, name='x_mask')
        self.t_mask   = tf.placeholder(tf.float32, shape=shape, name='t_mask')

        shape = [None, None]
        self.X_spaces     = tf.placeholder(tf.int32, shape=shape,  name='X_spaces')
        self.X_spaces_len = tf.placeholder(tf.int32, shape=[None], name='X_spaces_len')

        self.num_samples = tf.placeholder(tf.int32, shape=1, name='num_samples')

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

        def encoder(inputs, lengths, name, reverse=False):
            with tf.variable_scope(name):
                weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
                input_units = inputs.get_shape()[2]
                W_z = tf.get_variable('W_z',
                                      shape=[input_units+self.rnn_units, self.rnn_units],
                                      initializer=weight_initializer)
                W_r = tf.get_variable('W_r',
                                      shape=[input_units+self.rnn_units, self.rnn_units],
                                      initializer=weight_initializer)
                W_h = tf.get_variable('W_h',
                                      shape=[input_units+self.rnn_units, self.rnn_units],
                                      initializer=weight_initializer)
                b_z = tf.get_variable('b_z',
                                      shape=[self.rnn_units],
                                      initializer=tf.constant_initializer(1.0))
                b_r = tf.get_variable('b_r',
                                      shape=[self.rnn_units],
                                      initializer=tf.constant_initializer(1.0))
                b_h = tf.get_variable('b_h',
                                      shape=[self.rnn_units],
                                      initializer=tf.constant_initializer())

                max_sequence_length = tf.reduce_max(lengths)
                min_sequence_length = tf.reduce_min(lengths)

                time = tf.constant(0)

                state_shape = tf.concat(0, [tf.expand_dims(tf.shape(lengths)[0], 0),
                                            tf.expand_dims(tf.constant(self.rnn_units), 0)])
                # state_shape = tf.Print(state_shape, [state_shape])
                state = tf.zeros(state_shape, dtype=tf.float32)

                if reverse:
                    inputs = tf.reverse(inputs, dims=[False, True, False])
                inputs = tf.transpose(inputs, perm=[1, 0, 2])
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
                        if reverse:
                            return tf.select(tf.less(max_sequence_length-lengths, time), old_state, new_state)
                        else:
                            return tf.select(tf.less(time, lengths), new_state, old_state)

                    if reverse:
                        state = tf.cond(
                            tf.greater(time, max_sequence_length-min_sequence_length),
                            updateall,
                            updatesome
                        )
                    else:
                        state = tf.cond(tf.less(time, min_sequence_length), updateall, updatesome)

                    return (time + 1, state, output_ta_t)

                loop_vars = [time, state, output_ta]

                time, state, output_ta = tf.while_loop(encoder_cond, encoder_body, loop_vars)

                enc_state = state
                enc_out = tf.transpose(output_ta.pack(), perm=[1, 0, 2])

                if reverse:
                    tf.reverse(enc_out, dims=[False, True, False])

                return enc_state, enc_out

        char_enc_state, char_enc_out = encoder(X_embedded, self.X_len, 'char_encoder')
        # reshape worded output to reflect per sentence
        shape = tf.pack([tf.squeeze(self.num_samples), -1, self.rnn_units])
        char_enc_out = tf.reshape(char_enc_state, shape)
        char_enc_out.set_shape([None, None, self.rnn_units])

        # compute no. spaces in sentence from no. words/tokens in
        shape = tf.pack([tf.squeeze(self.num_samples), -1])
        num_spaces = tf.reshape(self.X_len, shape)
        num_spaces = tf.greater(num_spaces, 0)
        num_spaces = tf.to_int32(num_spaces)
        num_spaces = tf.reduce_sum(num_spaces, reduction_indices=1)

        #char2word = _grid_gather(char_enc_out, self.X_spaces)
        #char2word.set_shape([None, None, self.rnn_units])
        #word_enc_state, word_enc_out = encoder(char2word, self.X_spaces_len, 'word_encoder')
        word_enc_state, word_enc_out = encoder(char_enc_out, num_spaces, 'word_encoder')

        with tf.variable_scope('decoder'):
            weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
            W_z = tf.get_variable('W_z',
                                  shape=[self.embedd_dims+self.rnn_units*2, self.rnn_units],
                                  initializer=weight_initializer)
            W_r = tf.get_variable('W_r',
                                  shape=[self.embedd_dims+self.rnn_units*2, self.rnn_units],
                                  initializer=weight_initializer)
            W_h = tf.get_variable('W_h',
                                  shape=[self.embedd_dims+self.rnn_units*2, self.rnn_units],
                                  initializer=weight_initializer)
            b_z = tf.get_variable('b_z',
                                  shape=[self.rnn_units],
                                  initializer=tf.constant_initializer(1.0))
            b_r = tf.get_variable('b_r',
                                  shape=[self.rnn_units],
                                  initializer=tf.constant_initializer(1.0))
            b_h = tf.get_variable('b_h',
                                  shape=[self.rnn_units],
                                  initializer=tf.constant_initializer())

            # for attention
            attn_units = self.rnn_units
            attn_len = tf.shape(word_enc_out)[1]
            attn_size = self.rnn_units
            W_a = tf.get_variable('W_a',
                                  shape=[self.rnn_units, attn_units],
                                  initializer=weight_initializer)
            U_a = tf.get_variable('U_a',
                                  shape=[1, 1, attn_size, attn_units],
                                  initializer=weight_initializer)
            b_a = tf.get_variable('b_a',
                                  shape=[attn_units],
                                  initializer=tf.constant_initializer())
            v_a = tf.get_variable('v_a',
                                  shape=[attn_units],
                                  initializer=weight_initializer)

            # TODO: don't use convolutions!
            # TODO: fix the bias (b_a)
            hidden = tf.reshape(word_enc_out, tf.pack([-1, attn_len, 1, attn_size]))
            part1 = tf.nn.conv2d(hidden, U_a, [1, 1, 1, 1], "SAME")
            part1 = tf.squeeze(part1, [2]) # squeeze over third dimension
            max_sequence_length = tf.reduce_max(self.t_len)

            time = tf.constant(0)

            state = word_enc_state

            inputs = tf.transpose(t_embedded, perm=[1, 0, 2])
            input_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)
            input_ta = input_ta.unpack(inputs)

            output_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)

            def decoder_cond(time, state, output_ta_t):
                return tf.less(time, max_sequence_length)

            def decoder_body_builder(feedback=False):
                def decoder_body(time, old_state, output_ta_t):
                    if feedback:
                        prev_1 = tf.matmul(old_state, W_out) + b_out
                        x_t = tf.gather(self.embeddings, tf.argmax(prev_1, 1))
                    else:
                        x_t = input_ta.read(time)

                    # attention
                    part2 = tf.matmul(old_state, W_a) + b_a
                    part2 = tf.expand_dims(part2, 1)
                    john = part1 + part2
                    e = tf.reduce_sum(v_a * tf.tanh(john), [2])
                    alpha = tf.nn.softmax(e)
                    c = tf.reduce_sum(tf.expand_dims(alpha, 2) * tf.squeeze(hidden), [1])
                    # TODO: insert improved masking

                    # GRU
                    con = tf.concat(1, [x_t, old_state, c])
                    z = tf.sigmoid(tf.matmul(con, W_z) + b_z)
                    r = tf.sigmoid(tf.matmul(con, W_r) + b_r)
                    con = tf.concat(1, [x_t, r*old_state, c])
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
        out_shape = tf.concat(0, [self.num_samples,
                                  tf.expand_dims(max_sequence_length, 0),
                                  tf.expand_dims(tf.constant(self.alphabet_size), 0)])
        self.out_tensor = tf.reshape(out_tensor, out_shape)
        self.out_tensor.set_shape([None, None, self.alphabet_size])

        valid_out_tensor = tf.reshape(valid_dec_out, [-1, self.rnn_units])
        valid_out_tensor = tf.matmul(valid_out_tensor, W_out) + b_out
        self.valid_out_tensor = tf.reshape(valid_out_tensor, out_shape)

        self.out = None

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
        self.batch_generator['train'] = tl.WordedTextBatchGenerator(
            loader=train_loader,
            batch_size=self.batch_size,
            use_dynamic_array_sizes=True,
            **self.schedule_kwargs)

        # load validation set
        print('Load validation set')
        valid_loader = tl.TextLoader(paths_X=self.valid_x_files,
                                     paths_t=self.valid_t_files,
                                     seq_len=self.seq_len)
        self.batch_generator['valid'] = tl.WordedTextBatchGenerator(
            loader=valid_loader,
            batch_size=self.batch_size,
            use_dynamic_array_sizes=True)

    def valid_dict(self, batch, feedback=True):
        """ Return feed_dict for validation """
        return { self.Xs:     batch['x_encoded'],
                 self.ts:     batch['t_encoded'],
                 self.ts_go:  batch['t_encoded_go'],
                 self.X_len:  batch['x_len'],
                 self.t_len:  batch['t_len'],
                 #self.x_mask: batch['x_mask'],
                 self.t_mask: batch['t_mask'],
                 self.num_samples: batch['num_samples'],
                 self.feedback: feedback,
                 self.X_spaces: batch['x_spaces'],
                 self.X_spaces_len: batch['x_spaces_len'] }
