import tensorflow as tf
import numpy as np
from tensorflow.python.ops import seq2seq
import rnn_cell
#from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn

import text_loader as tl


class Model(object):
    # settings that affect train.py
    batch_size = 64
    seq_len = 50
    name = None  # (string) For saving logs and checkpoints. (None to disable.)
    visualize_freq = 1000  # Visualize training X, y, and t. (0 to disable.)
    log_freq = 10  # How often to print updates during training.
    save_freq = 0  # How often to save checkpoints. (0 to disable.)
    valid_freq = 100  # How often to validate.
    iterations = 32000  # How many iterations to train for before stopping.
    warmup = 100  # How many iterations to warm up for.
    train_feedback = False  # Enable feedback during training?
    tb_log_freq = 100  # How often to save logs for TensorBoard

    # datasets
    train_x_files = ['data/train/europarl-v7.da-en.en']
    train_t_files = ['data/train/europarl-v7.da-en.da']
    valid_x_files = ['data/test/devtest2006.en', 'data/test/test2006.en']
    valid_t_files = ['data/test/devtest2006.da', 'data/test/test2006.da']

    # settings that are local to the model
    alphabet_size = 337
    rnn_units = 400
    embedd_dims = 16
    learning_rate = 0.001
    reg_scale = 0.0001
    clip_norm = 1

    swap_schedule = {
            0: 0.0,
#            5000: 0.05,
#            10000: 0.1,
#            20000: 0.15,
#            30000: 0.25,
#            40000: 0.3,
#            50000: 0.35,
#            60000: 0.39,
            }

    def __init__(self, Xs, X_len, ts, ts_go, t_mask, feedback, X_spaces,
                 X_spaces_len):
        self.max_x_seq_len = self.seq_len
        self.max_t_seq_len = self.seq_len

        self.Xs, self.X_len, self.feedback = Xs, X_len, feedback
        self.ts, self.ts_go, self.t_mask = ts, ts_go, t_mask
        self.X_spaces = X_spaces
        self.X_spaces_len = X_spaces_len

        self.build()
        self.build_loss()
        self.build_prediction()
        self.training()

    def build(self):
        print('Building model')
        self.embeddings = tf.Variable(
            tf.random_normal([self.alphabet_size, self.embedd_dims],
            stddev=0.1), name='embeddings')

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
            W_out = tf.get_variable('W_out', [self.rnn_units, self.alphabet_size])
            b_out = tf.get_variable('b_out', [self.alphabet_size])

        cell = rnn_cell.GRUCell(self.rnn_units)

        # encoder
        enc_outputs, enc_state = rnn.rnn(
            cell=cell,
            inputs=X_list,
            dtype=tf.float32,
            sequence_length=self.X_len,
            scope='rnn_encoder')

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

    def build_loss(self):
        """Build a loss function and accuracy for the model."""
        print('Building model loss and accuracy')

        with tf.variable_scope('accuracy'):
            argmax = tf.to_int32(tf.argmax(self.out_tensor, 2))
            correct = tf.to_float(tf.equal(argmax, self.ts)) * self.t_mask
            self.accuracy = tf.reduce_sum(correct) / tf.reduce_sum(self.t_mask)

            tf.scalar_summary('accuracy', self.accuracy)

        with tf.variable_scope('loss'):
            with tf.variable_scope('split_t_and_mask'):
                # Why are we overwriting these values?
                ts = tf.split(split_dim=1,
                              num_split=self.max_t_seq_len,
                              value=self.ts)

                t_mask = tf.split(
                    split_dim=1,
                    num_split=self.max_t_seq_len,
                    value=self.t_mask)

                t_mask = [tf.squeeze(weight) for weight in t_mask]

            loss = seq2seq.sequence_loss(
                self.out,
                ts,
                t_mask,
                self.max_t_seq_len)

            with tf.variable_scope('regularization'):
                regularize = tf.contrib.layers.l2_regularizer(self.reg_scale)
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

    def training(self):
        print('Building model training')

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Do gradient clipping
        # NOTE: this is the correct, but slower clipping by global norm.
        # Maybe it's worth trying the faster tf.clip_by_norm()
        # (See the documentation for tf.clip_by_global_norm() for more info)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        grads, variables = zip(*grads_and_vars)  # unzip list of tuples
        clipped_grads, global_norm = tf.clip_by_global_norm(grads,
                                                            self.clip_norm)
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

    def setup_batch_generators(self):
        """Load the datasets"""
        batch_generator = dict()

        # load training set
        print('\nload training set')
        train_loader = tl.TextLoader(
            paths_X=self.train_x_files,
            paths_t=self.train_t_files,
            seq_len=self.seq_len
        )
        train_iteration_schedule = tl.BucketIterationSchedule(shuffle=True, repeat=True)
        batch_generator['train'] = tl.TextBatchGenerator(
            loader=train_loader,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            iteration_schedule=train_iteration_schedule
        )

        # load validation set
        print('\nload validation set')
        valid_loader = tl.TextLoader(
            paths_X=self.valid_x_files,
            paths_t=self.valid_t_files,
            seq_len=self.seq_len
        )
        batch_generator['valid'] = tl.TextBatchGenerator(
            loader=valid_loader,
            batch_size=self.batch_size,
            seq_len=self.seq_len
        )

        return batch_generator
