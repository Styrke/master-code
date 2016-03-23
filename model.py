import tensorflow as tf
import numpy as np
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
from utils.tfextensions import grid_gather


class Model(object):

    def __init__(self, alphabet_size, embedd_dims=16, max_x_seq_len=25,
            max_t_seq_len=25):
        self.alphabet_size = alphabet_size
        self.embedd_dims = embedd_dims
        self.max_x_seq_len = max_x_seq_len
        self.max_t_seq_len = max_t_seq_len
        # rnn output size must equal alphabet size for decoder feedback to work
        self.rnn_units = 400

    def build(self, Xs, X_len, ts, feedback):
        print('Building model')

        self.embeddings = tf.Variable(
            tf.random_uniform([self.alphabet_size, self.embedd_dims]),
            name='embeddings')

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

        # The loop function provides inputs to the decoder:
        def decoder_loop_function(prev, i):
            def feedback_on():
                prev_1 = tf.matmul(prev, W_out) + b_out
                # feedback is on, so feed the decoder with the previous output
                return tf.gather(self.embeddings, tf.argmax(prev_1, 1))

            def feedback_off():
                # feedback is off, so just feed the decoder with t's
                return t_list[i]

            return tf.cond(feedback, feedback_on, feedback_off)

        # decoder
        dec_out, dec_state = seq2seq.rnn_decoder(
            decoder_inputs=t_list,
            initial_state=enc_state,
            cell=cell,
            loop_function=decoder_loop_function)

        self.out = []
        for d in dec_out:
            self.out.append(tf.matmul(d, W_out) + b_out)

        # for debugging network
        out_packed = tf.pack(self.out)
        out_packed = tf.transpose(out_packed, perm=[1, 0, 2])
        print(out_packed.get_shape())
        self.out_tensor = out_packed

        # add TensorBoard summaries for all variables
        tf.contrib.layers.summarize_variables()

    def build_loss(self, ts, t_mask, reg_scale=0.0001):
        """Build a loss function and accuracy for the model.

        Keyword arguments:
        ts -- targets to predict
        t_mask -- mask for the targets
        reg_scale -- regularization scale in range [0.0, 1.0]. 0.0 to
            disable.
        """
        print('Building model loss and accuracy')

        with tf.variable_scope('accuracy'):
            argmax = tf.to_int32(tf.argmax(self.out_tensor, 2))
            correct = tf.to_float(tf.equal(argmax, ts)) * t_mask
            self.accuracy = tf.reduce_sum(correct) / tf.reduce_sum(t_mask)

            tf.scalar_summary('accuracy', self.accuracy)

        with tf.variable_scope('loss'):
            with tf.variable_scope('split_t_and_mask'):
                ts = tf.split(split_dim=1,
                              num_split=self.max_t_seq_len,
                              value=ts)

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

            with tf.variable_scope('regularization'):
                regularize = tf.contrib.layers.l2_regularizer(reg_scale)
                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                reg_term = sum([regularize(param) for param in params])

            loss = loss + reg_term

            tf.scalar_summary('loss', loss)

        self.loss = loss

    def build_prediction(self):
        with tf.variable_scope('prediction'):
            # logits is a list of tensors of shape [batch_size,
            # alphabet_size]. We need a tensor shape [batch_size,
            # target_seq_len, alphabet_size]
            packed_logits = tf.transpose(tf.pack(self.out), perm=[1, 0, 2])

            self.ys = tf.argmax(packed_logits, dimension=2)


    def training(self, learning_rate, clip_norm=1):
        print('Building model training')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer()

        # Do gradient clipping
        # NOTE: this is the correct, but slower clipping by global norm.
        # Maybe it's worth trying the faster tf.clip_by_norm()
        # (See the documentation for tf.clip_by_global_norm() for more info)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        grads, variables = zip(*grads_and_vars)  # unzip list of tuples
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm)
        clipped_grads_and_vars = zip(clipped_grads, variables)

        # Create TensorBoard scalar summary for global gradient norm
        tf.scalar_summary('global gradient norm', global_norm)

        # Create TensorBoard summaries for gradients
        for grad, var in grads_and_vars:
            # Sparse tensor updates can't be summarized, so avoid doing that:
            if isinstance(grad, tf.Tensor):
                tf.histogram_summary('grad_' + var.name, grad)

        # make training op for applying the gradients
        self.train_op = optimizer.apply_gradients(clipped_grads_and_vars,
                                                  global_step=self.global_step)
