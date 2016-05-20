import tensorflow as tf
import numpy as np
from tensorflow.python.ops import seq2seq

import rnn_custom as rnn
#from tensorflow.python.ops import rnn

# NOTE swap these two lines to use custom or TF's `rnn_cell`
import custom_gru_cell as rnn_cell
#from tensorflow.python.ops import rnn_cell

import text_loader as tl


class Model(object):
    # settings that affect train.py
    batch_size = 64
    seq_len = 50
    name = None  # (string) For saving logs and checkpoints. (None to disable.)
    visualize_freq = 1000  # Visualize training X, y, and t. (0 to disable.)
    log_freq = 10  # How often to print updates during training.
    save_freq = 0  # How often to save checkpoints. (0 to disable.)
    valid_freq = 500  # How often to validate.
    iterations = 5*32000  # How many iterations to train for before stopping.
    train_feedback = False  # Enable feedback during training?
    tb_log_freq = 100  # How often to save logs for TensorBoard

    # datasets
    train_x_files = ['data/train/europarl-v7.da-en.en']
    train_t_files = ['data/train/europarl-v7.da-en.da']
    valid_x_files = ['data/valid/devtest2006.en', 'data/valid/test2006.en',
                     'data/valid/test2007.en', 'data/valid/test2008.en']
    valid_t_files = ['data/valid/devtest2006.da', 'data/valid/test2006.da',
                     'data/valid/test2007.da', 'data/valid/test2008.da']

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

    # kwargs for scheduling function
    schedule_kwargs = {
            'warmup_iterations': 100,  # if warmup_schedule is used
            'warmup_function':  None,  # if warmup_schedule is used
            'regular_function': None,  # if warmup_schedule is used
            'shuffle': True,
            'repeat':  True,
            'sort':    False,
            'fuzzyness': 3
            }

    def __init__(self):
        self.max_x_seq_len = self.max_t_seq_len = self.seq_len

        # TF placeholders
        self.setup_placeholders()

        # schedule functions
        self.train_schedule_function = tl.warmup_schedule
        self.valid_schedule_function = None # falls back to frostings.default_schedule

        print("Model instantiation");
        self.build()
        self.loss, self.accuracy = self.build_loss(self.out, self.out_tensor)
        self.valid_loss, self.valid_accuracy = self.build_valid_loss()
        self.ys = self.build_prediction(self.out_tensor)
        self.valid_ys = self.build_valid_prediction()
        self.build_training()

        # Create TensorBoard scalar summaries
        tf.scalar_summary('train/loss', self.loss)
        tf.scalar_summary('train/accuracy', self.accuracy)

        # setup batch generators
        self.setup_batch_generators()


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

        with tf.variable_scope('split_X_inputs'):
            X_list = tf.split(split_dim=1,
                              num_split=self.max_x_seq_len,
                              value=X_embedded)
            X_list = [tf.squeeze(X) for X in X_list]
            [X.set_shape([None, self.embedd_dims]) for X in X_list]

        with tf.variable_scope('split_t_inputs'):
            t_list = tf.split(split_dim=1,
                              num_split=self.max_t_seq_len,
                              value=t_embedded)
            t_list = [tf.squeeze(t) for t in t_list]
            [t.set_shape([None, self.embedd_dims]) for t in t_list]

        with tf.variable_scope('dense_out'):
            W_out = tf.get_variable('W_out', [self.rnn_units, self.alphabet_size])
            b_out = tf.get_variable('b_out', [self.alphabet_size])

        cell = rnn_cell.GRUCell(self.rnn_units)

        # encoder
        enc_outputs, enc_state = rnn.rnn(cell=cell,
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
        dec_out, dec_state = (
                seq2seq.rnn_decoder(decoder_inputs=t_list,
                                    initial_state=enc_state,
                                    cell=cell,
                                    loop_function=decoder_loop_function) )

        self.out = [tf.matmul(d, W_out) + b_out for d in dec_out]

        # for debugging network (NOTE should write this outside of build)
        out_packed = tf.pack(self.out)
        out_packed = tf.transpose(out_packed, perm=[1, 0, 2])
        self.out_tensor = out_packed

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
            with tf.variable_scope('split_t_and_mask'):
                split_kwargs = { 'split_dim': 1,
                                 'num_split': self.max_t_seq_len }
                ts     = tf.split(value=self.ts,     **split_kwargs)
                t_mask = tf.split(value=self.t_mask, **split_kwargs)
                t_mask = [tf.squeeze(weight) for weight in t_mask]

            loss = seq2seq.sequence_loss(out, ts, t_mask,
                                         self.max_t_seq_len)

            with tf.variable_scope('regularization'):
                regularize = tf.contrib.layers.l2_regularizer(self.reg_scale)
                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                reg_term = sum([regularize(param) for param in params])

            loss += reg_term

        return loss, accuracy

    def build_valid_loss(self):
        return self.loss, self.accuracy

    def build_prediction(self, out_tensor):
        print('  Building prediction')
        with tf.variable_scope('prediction'):
            # logits is a list of tensors of shape [batch_size, alphabet_size].
            # We need shape of [batch_size, target_seq_len, alphabet_size].
            return tf.argmax(out_tensor, dimension=2)

    def build_valid_prediction(self):
        return self.ys

    def build_training(self):
        print('  Building training')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Do gradient clipping
        # NOTE: this is the correct, but slower clipping by global norm.
        # Maybe it's worth trying the faster tf.clip_by_norm()
        # (See the documentation for tf.clip_by_global_norm() for more info)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
        clipped_gradients, global_norm = (
                tf.clip_by_global_norm(gradients, self.clip_norm) )
        clipped_grads_and_vars = zip(clipped_gradients, variables)

        # Create TensorBoard scalar summary for global gradient norm
        tf.scalar_summary('train/global gradient norm', global_norm)

        # Create TensorBoard summaries for gradients
        # for grad, var in grads_and_vars:
        #     # Sparse tensor updates can't be summarized, so avoid doing that:
        #     if isinstance(grad, tf.Tensor):
        #         tf.histogram_summary('grad_' + var.name, grad)

        # make training op for applying the gradients
        self.train_op = optimizer.apply_gradients(clipped_grads_and_vars,
                                                  global_step=self.global_step)

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
            **self.schedule_kwargs)

        # load validation set
        print('Load validation set')
        valid_loader = tl.TextLoader(paths_X=self.valid_x_files,
                                     paths_t=self.valid_t_files,
                                     seq_len=self.seq_len)
        self.batch_generator['valid'] = tl.TextBatchGenerator(
            loader=valid_loader,
            batch_size=self.batch_size)

    def valid_dict(self, batch, feedback=True):
        """ Return feed_dict for validation """
        return { self.Xs:     batch['x_encoded'],
                 self.ts:     batch['t_encoded'],
                 self.ts_go:  batch['t_encoded_go'],
                 self.X_len:  batch['x_len'],
                 self.t_len:  batch['t_len'],
                 self.t_mask: batch['t_mask'],
                 self.feedback: feedback,
                 self.X_spaces: batch['x_spaces'],
                 self.X_spaces_len: batch['x_spaces_len'] }

    def train_dict(self, batch):
        """ Return feed_dict for training.
        Reuse validation feed_dict because the only difference is feedback.
        """
        return self.valid_dict(batch, feedback=False)

    def build_feed_dict(self, batch, validate=False):
        return self.valid_dict(batch) if validate else self.train_dict(batch)

    def get_generator(self, validate=False):
        k = 'valid' if validate else 'train'
        return self.batch_generator[k].gen_batch

    def next_train_feed(self):
        generator = self.get_generator()
        for t_batch in generator(self.train_schedule_function):
            extra = { 't_len': t_batch['t_len'] }
            yield (self.build_feed_dict(t_batch), extra)

    def next_valid_feed(self):
        generator = self.get_generator(validate=True)
        for v_batch in generator(self.valid_schedule_function):
            yield self.build_feed_dict(v_batch, validate=True)

    def get_alphabet(self):
        return self.batch_generator['train'].alphabet
