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

    def build(self):
        print('Building model')
        self.embeddings = tf.Variable(
            tf.random_normal([self.alphabet_size, self.embedd_dims],
            stddev=0.1), name='embeddings')

        self.t_embeddings = tf.Variable(
            tf.random_normal([self.alphabet_size, self.embedd_dims],
            stddev=0.1), name='embeddings')

        X_embedded = tf.gather(self.embeddings, self.Xs, name='embed_X')
        t_embedded = tf.gather(self.t_embeddings, self.ts_go, name='embed_t')

        with tf.variable_scope('dense_out'):
            W_out = tf.get_variable('W_out', [self.rnn_units*2, self.alphabet_size])
            b_out = tf.get_variable('b_out', [self.alphabet_size])

        # forward encoding
        char_enc_state, char_enc_out = encoder(X_embedded, self.X_len, 'char_encoder', self.rnn_units)
        char2word = _grid_gather(char_enc_out, self.X_spaces)
        char2word.set_shape([None, None, self.rnn_units])
        word_enc_state, word_enc_out = encoder(char2word, self.X_spaces_len, 'word_encoder', self.rnn_units)

        # backward encoding words
        char2word = tf.reverse_sequence(char2word, tf.to_int64(self.X_spaces_len), 1)
        char2word.set_shape([None, None, self.rnn_units])
        word_enc_state_bck, word_enc_out_bck = encoder(char2word, self.X_spaces_len, 'word_encoder_backwards', self.rnn_units)
        word_enc_out_bck = tf.reverse_sequence(word_enc_out_bck, tf.to_int64(self.X_spaces_len), 1)

        word_enc_state = tf.concat(1, [word_enc_state, word_enc_state_bck])
        word_enc_out = tf.concat(2, [word_enc_out, word_enc_out_bck])

        # decoding
        dec_state, dec_out, valid_dec_out = decoder(word_enc_out, self.X_spaces_len, word_enc_state,
                                                    t_embedded, self.t_len, self.rnn_units*2,
                                                    self.rnn_units, self.t_embeddings, W_out, b_out)

        out_tensor = tf.reshape(dec_out, [-1, self.rnn_units*2])
        out_tensor = tf.matmul(out_tensor, W_out) + b_out
        out_shape = tf.concat(0, [tf.expand_dims(tf.shape(self.X_len)[0], 0),
                                  tf.expand_dims(tf.shape(t_embedded)[1], 0),
                                  tf.expand_dims(tf.constant(self.alphabet_size), 0)])
        self.out_tensor = tf.reshape(out_tensor, out_shape)
        self.out_tensor.set_shape([None, None, self.alphabet_size])

        valid_out_tensor = tf.reshape(valid_dec_out, [-1, self.rnn_units*2])
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

    def valid_dict(self, batch, feedback=True):
        """ Return feed_dict for validation """
        return { self.Xs:     batch['x_encoded'],
                 self.ts:     batch['t_encoded'],
                 self.ts_go:  batch['t_encoded_go'],
                 self.X_len:  batch['x_len'],
                 self.t_len:  batch['t_len'],
                 self.x_mask: batch['x_mask'],
                 self.t_mask: batch['t_mask'],
                 self.feedback: feedback,
                 self.X_spaces: batch['x_spaces'],
                 self.X_spaces_len: batch['x_spaces_len'] }


def encoder(inputs, lengths, name, num_units, reverse=False):
    with tf.variable_scope(name):
        weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
        input_units = inputs.get_shape()[2]
        W_z = tf.get_variable('W_z',
                              shape=[input_units+num_units, num_units],
                              initializer=weight_initializer)
        W_r = tf.get_variable('W_r',
                              shape=[input_units+num_units, num_units],
                              initializer=weight_initializer)
        W_h = tf.get_variable('W_h',
                              shape=[input_units+num_units, num_units],
                              initializer=weight_initializer)
        b_z = tf.get_variable('b_z',
                              shape=[num_units],
                              initializer=tf.constant_initializer(1.0))
        b_r = tf.get_variable('b_r',
                              shape=[num_units],
                              initializer=tf.constant_initializer(1.0))
        b_h = tf.get_variable('b_h',
                              shape=[num_units],
                              initializer=tf.constant_initializer())

        max_sequence_length = tf.reduce_max(lengths)
        min_sequence_length = tf.reduce_min(lengths)

        time = tf.constant(0)

        state_shape = tf.concat(0, [tf.expand_dims(tf.shape(lengths)[0], 0),
                                    tf.expand_dims(tf.constant(num_units), 0)])
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
                    return tf.select(
                        tf.greater_equal(time, max_sequence_length-lengths),
                        new_state,
                        old_state)
                else:
                    return tf.select(tf.less(time, lengths), new_state, old_state)

            if reverse:
                state = tf.cond(
                    tf.greater_equal(time, max_sequence_length-min_sequence_length),
                    updateall,
                    updatesome)
            else:
                state = tf.cond(tf.less(time, min_sequence_length), updateall, updatesome)

            return (time + 1, state, output_ta_t)

        loop_vars = [time, state, output_ta]

        time, state, output_ta = tf.while_loop(encoder_cond, encoder_body, loop_vars)

        enc_state = state
        enc_out = tf.transpose(output_ta.pack(), perm=[1, 0, 2])

        if reverse:
            enc_out = tf.reverse(enc_out, dims=[False, True, False])

        enc_out.set_shape([None, None, num_units])

        return enc_state, enc_out

def decoder(attention_input, attention_lengths, initial_state, target_input, target_input_lengths, num_units,
            num_attn_units, embeddings, W_out, b_out, name='decoder'):
    """
    Keyword arguments:
        attention_input:    the input to put attention on. expected dims: [batch_size, attention_length, attention_dims]
        initial_state:      The initial state for the decoder RNN.
        target_input:       The target to replicate. Expected: [batch_size, max_target_sequence_len, embedding_dims]
        num_units:          Number of GRU units in the decoder.
        num_attn_units:     Number of units in the alignment layer that produces the context vectors.
    """
    with tf.variable_scope(name):
        target_dims = target_input.get_shape()[2]
        attention_dims = attention_input.get_shape()[2]
        attn_len = tf.shape(attention_input)[1]
        max_sequence_length = tf.reduce_max(target_input_lengths)

        weight_initializer = tf.truncated_normal_initializer(stddev=0.1)
        W_z = tf.get_variable('W_z',
                              shape=[target_dims+num_units*2, num_units],
                              initializer=weight_initializer)
        W_r = tf.get_variable('W_r',
                              shape=[target_dims+num_units*2, num_units],
                              initializer=weight_initializer)
        W_h = tf.get_variable('W_h',
                              shape=[target_dims+num_units*2, num_units],
                              initializer=weight_initializer)
        b_z = tf.get_variable('b_z',
                              shape=[num_units],
                              initializer=tf.constant_initializer(1.0))
        b_r = tf.get_variable('b_r',
                              shape=[num_units],
                              initializer=tf.constant_initializer(1.0))
        b_h = tf.get_variable('b_h',
                              shape=[num_units],
                              initializer=tf.constant_initializer())

        # for attention
        W_a = tf.get_variable('W_a',
                              shape=[attention_dims, num_attn_units],
                              initializer=weight_initializer)
        U_a = tf.get_variable('U_a',
                              shape=[1, 1, attention_dims, num_attn_units],
                              initializer=weight_initializer)
        b_a = tf.get_variable('b_a',
                              shape=[num_attn_units],
                              initializer=tf.constant_initializer())
        v_a = tf.get_variable('v_a',
                              shape=[num_attn_units],
                              initializer=weight_initializer)

        # TODO: don't use convolutions!
        # TODO: fix the bias (b_a)
        hidden = tf.reshape(attention_input, tf.pack([-1, attn_len, 1, attention_dims]))
        part1 = tf.nn.conv2d(hidden, U_a, [1, 1, 1, 1], "SAME")
        part1 = tf.squeeze(part1, [2])  # squeeze out the third dimension

        inputs = tf.transpose(target_input, perm=[1, 0, 2])
        input_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)
        input_ta = input_ta.unpack(inputs)

        def decoder_cond(time, state, output_ta_t):
            return tf.less(time, max_sequence_length)

        def decoder_body_builder(feedback=False):
            def decoder_body(time, old_state, output_ta_t):
                if feedback:
                    def from_previous():
                        prev_1 = tf.matmul(old_state, W_out) + b_out
                        return tf.gather(embeddings, tf.argmax(prev_1, 1))
                    x_t = tf.cond(tf.greater(time, 0), from_previous, lambda: input_ta.read(0))
                else:
                    x_t = input_ta.read(time)

                # attention
                part2 = tf.matmul(old_state, W_a) + b_a
                part2 = tf.expand_dims(part2, 1)
                john = part1 + part2
                e = tf.reduce_sum(v_a * tf.tanh(john), [2])
                alpha = tf.nn.softmax(e)
                alpha = tf.to_float(mask(attention_lengths)) * alpha
                alpha = alpha / tf.reduce_sum(alpha, [1], keep_dims=True)
                c = tf.reduce_sum(tf.expand_dims(alpha, 2) * tf.squeeze(hidden), [1])

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


        output_ta = tensor_array_ops.TensorArray(tf.float32, size=1, dynamic_size=True)
        time = tf.constant(0)
        loop_vars = [time, initial_state, output_ta]

        _, state, output_ta = tf.while_loop(decoder_cond,
                                            decoder_body_builder(),
                                            loop_vars)
        _, valid_state, valid_output_ta = tf.while_loop(decoder_cond,
                                                        decoder_body_builder(feedback=True),
                                                        loop_vars)

        dec_state = state
        dec_out = tf.transpose(output_ta.pack(), perm=[1, 0, 2])
        valid_dec_out = tf.transpose(valid_output_ta.pack(), perm=[1, 0, 2])

        return dec_state, dec_out, valid_dec_out
