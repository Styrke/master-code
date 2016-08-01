import tensorflow as tf
import text_loader as tl
from utils.tfextensions import sequence_loss_tensor
from utils.tfextensions import _grid_gather
from utils.tfextensions import mask
from utils.rnn import encoder
from utils.rnn_hierachical import attention_decoder
from data.alphabet import Alphabet

class Model:
    # settings that affect train.py
    batch_size_train = 85000
    batch_size_valid = 128
    seq_len_x = 50
    seq_len_t = 50
    name = None  # (string) For saving logs and checkpoints. (None to disable.)
    visualize_freq = 10000  # Visualize training X, y, and t. (0 to disable.)
    log_freq = 100  # How often to print updates during training.
    save_freq = 1000  # How often to save checkpoints. (0 to disable.)
    valid_freq = 500  # How often to validate.
    iterations = 5*32000  # How many iterations to train for before stopping.
    train_feedback = False  # Enable feedback during training?
    tb_log_freq = 500  # How often to save logs for TensorBoard
    max_to_keep = 100

    # datasets
    #train_x_files = ['data/train/europarl-v7.de-en.en']
    #train_t_files = ['data/train/europarl-v7.de-en.de']
    train_x_files = ['data/train/europarl-v7.de-en.en.tok',
                     'data/train/commoncrawl.de-en.en.tok',
                     'data/train/news-commentary-v10.de-en.en.tok']
    train_t_files = ['data/train/europarl-v7.de-en.de.tok',
                     'data/train/commoncrawl.de-en.de.tok',
                     'data/train/news-commentary-v10.de-en.de.tok']
    #valid_x_files = ['data/valid/devtest2006.en', 'data/valid/test2006.en',
    #                 'data/valid/test2007.en', 'data/valid/test2008.en']
    #valid_t_files = ['data/valid/devtest2006.de', 'data/valid/test2006.de',
    #                 'data/valid/test2007.de', 'data/valid/test2008.de']
    valid_x_files = ['data/valid/newstest2013.en.tok']
    valid_t_files = ['data/valid/newstest2013.de.tok']
    test_x_files = ['data/valid/newstest2014.deen.en.tok']
    test_t_files = ['data/valid/newstest2014.deen.de.tok']

    # settings that are local to the model
    num_h = 5
    h_degree = 3
    alphabet_src_size = 310  # size of alphabet
    alphabet_tar_size = 310  # size of alphabet
    alphabet_src = Alphabet('data/alphabet/dict_wmt_tok.de-en.en', eos='*')
    alphabet_tar = Alphabet('data/alphabet/dict_wmt_tok.de-en.de', eos='*', sos='')
    char_encoder_units = 300  # number of units in character-level encoder
    h_encoder_units = 300  # num nuits in word-level encoders (both forwards and back)
    attn_units = 200  # num units used for attention in the decoder.
    attn_rnn_units = 200  # num units used for attention in the decoder.
    embedd_dims = 256  # size of character embeddings
    learning_rate = 0.001
    reg_scale = 0.000001
    clip_norm = 1

    swap_schedule = {
        0: 0.0
    }

    # kwargs for scheduling function
    schedule_kwargs = {
        'fuzzyness': 3
    }

    def __init__(self):
        self.max_x_seq_len = self.seq_len_x
        self.max_t_seq_len = self.seq_len_t

        # TF placeholders
        self.setup_placeholders()

        # schedule functions
        self.train_schedule_function = tl.variable_bucket_schedule
        self.valid_schedule_function = None  # falls back to frostings.default_schedule
        self.test_schedule_function = None

        print("Model instantiation")
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
        # not including h[0] because char encoder is wierd with only forward n such.
        self.X_h      = dict()
        self.X_h_len  = dict()
        for i in range(self.num_h):
            self.X_h[i] = tf.placeholder(tf.int32, shape=shape,  name=('X_h%d' % i))
            self.X_h_len[i] = tf.placeholder(tf.int32, shape=[None], name=('X_h%d_len' % i))
        print('hi')
        print(self.X_h_len)

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
            W_out = tf.get_variable('W_out', [self.h_encoder_units*2, self.alphabet_tar_size])
            b_out = tf.get_variable('b_out', [self.alphabet_tar_size])

        # forward encoding
        char_enc_state, char_enc_out = encoder(X_embedded, self.X_len, 'char_encoder', self.char_encoder_units)
        char2word = _grid_gather(char_enc_out, self.X_h[0])
        char2word.set_shape([None, None, self.char_encoder_units])
        word_enc_state, word_enc_out = encoder(char2word, self.X_h_len[0], 'word_encoder', self.h_encoder_units)

        # backward encoding words
        char2word = tf.reverse_sequence(char2word, tf.to_int64(self.X_h_len[0]), 1)
        char2word.set_shape([None, None, self.char_encoder_units])
        word_enc_state_bck, word_enc_out_bck = encoder(char2word, self.X_h_len[0], 'word_encoder_backwards', self.h_encoder_units)
        word_enc_out_bck = tf.reverse_sequence(word_enc_out_bck, tf.to_int64(self.X_h_len[0]), 1)

        word_enc_state = tf.concat(1, [word_enc_state, word_enc_state_bck])
        word_enc_out = tf.concat(2, [word_enc_out, word_enc_out_bck])

        # h gather

        h_enc_state = dict()
        h_enc_state[0] = word_enc_state
        h_enc_out = dict()
        h_enc_out[0] = word_enc_out

        for i in range(1, self.num_h):
            gathered_out = _grid_gather(h_enc_out[(i-1)], self.X_h[i])
            gathered_out.set_shape([None, None, self.h_encoder_units*2])
            h_forward_state, h_forward_out = encoder(gathered_out, self.X_h_len[i], ("h%d_encoder" % i), self.h_encoder_units)
            gathered_out = tf.reverse_sequence(gathered_out, tf.to_int64(self.X_h_len[i]), 1)
            gathered_out.set_shape([None, None, self.h_encoder_units*2])
            h_backwards_state, h_backwards_out = encoder(gathered_out, self.X_h_len[i], ("h%d_encoder_backwards" % i), self.h_encoder_units)
            h_backwards_out = tf.reverse_sequence(h_backwards_out, tf.to_int64(self.X_h_len[i]), 1)
            h_enc_state[i] = tf.concat(1, [h_forward_state, h_backwards_state])
            h_enc_out[i] = tf.concat(2, [h_forward_out, h_backwards_out])


        # decoding
        dec_state, dec_out, valid_dec_out, *self.valid_a_list = (#self.valid_a, self.valid_az, self.valid_ar = (
            attention_decoder(h_input=h_enc_out,
                              h_lengths=self.X_h_len,
                              h_state=h_enc_state,
                              num_h=self.num_h,
                              target_input=t_embedded,
                              target_input_lengths=self.t_len,
                              num_attn_units=self.attn_units,
                              num_attn_rnn_units=self.attn_rnn_units,
                              embeddings=self.t_embeddings,
                              W_out=W_out,
                              b_out=b_out))
        self.valid_a_list = self.valid_a_list[0] # I know, it's hacked ...

        assert self.num_h in [1, 5]
        if self.num_h == 1:
            self.valid_a0 = self.valid_a_list[0]
            self.valid_az0 = self.valid_a_list[1]
            self.valid_ar0 = self.valid_a_list[2]

        if self.num_h == 5:
            self.valid_a0 = self.valid_a_list[0]
            self.valid_az0 = self.valid_a_list[1]
            self.valid_ar0 = self.valid_a_list[2]
            self.valid_a1 = self.valid_a_list[3]
            self.valid_az1 = self.valid_a_list[4]
            self.valid_ar1 = self.valid_a_list[5]
            self.valid_a2 = self.valid_a_list[6]
            self.valid_az2 = self.valid_a_list[7]
            self.valid_ar2 = self.valid_a_list[8]
            self.valid_a3 = self.valid_a_list[9]
            self.valid_az3 = self.valid_a_list[10]
            self.valid_ar3 = self.valid_a_list[11]
            self.valid_a4 = self.valid_a_list[12]
            self.valid_az4 = self.valid_a_list[13]
            self.valid_ar4 = self.valid_a_list[14]
        out_tensor = tf.reshape(dec_out, [-1, self.h_encoder_units*2]) # a hack for num_units
        out_tensor = tf.matmul(out_tensor, W_out) + b_out
        out_shape = tf.concat(0, [tf.expand_dims(tf.shape(self.X_len)[0], 0),
                                  tf.expand_dims(tf.shape(t_embedded)[1], 0),
                                  tf.expand_dims(tf.constant(self.alphabet_tar_size), 0)])

        self.out_tensor = tf.reshape(out_tensor, out_shape)
        self.out_tensor.set_shape([None, None, self.alphabet_tar_size])

        valid_out_tensor = tf.reshape(valid_dec_out, [-1, self.h_encoder_units*2])
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
                    self.alphabet_tar_size)

            with tf.variable_scope('regularization'):
                regularize = tf.contrib.layers.l2_regularizer(self.reg_scale)
                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                reg_term = sum([regularize(param) for param in params])

            loss += reg_term

        return loss, accuracy

    def build_valid_loss(self):
        return self.build_loss(self.out, self.valid_out_tensor)

    def build_prediction(self, out_tensor):
        print('  Building prediction')
        with tf.variable_scope('prediction'):
            # logits is a list of tensors of shape [batch_size, alphabet_size].
            # We need shape of [batch_size, target_seq_len, alphabet_size].
            return tf.argmax(out_tensor, dimension=2)

    def build_valid_prediction(self):
        return self.build_prediction(self.valid_out_tensor)

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
                                     seq_len_x=self.seq_len_x,
                                     seq_len_t=self.seq_len_t)
        self.batch_generator['train'] = tl.TextBatchGenerator(
            loader=train_loader,
            batch_size=self.batch_size_train,
            alphabet_src=self.alphabet_src,
            alphabet_tar=self.alphabet_tar,
            use_dynamic_array_sizes=True,
            num_h=self.num_h,
            h_degree=self.h_degree,
            **self.schedule_kwargs)

        # load validation set
        print('Load validation set')
        valid_loader = tl.TextLoader(paths_X=self.valid_x_files,
                                     paths_t=self.valid_t_files,
                                     seq_len_x=self.seq_len_x,
                                     seq_len_t=self.seq_len_t)
        self.batch_generator['valid'] = tl.TextBatchGenerator(
            loader=valid_loader,
            batch_size=self.batch_size_valid,
            alphabet_src=self.alphabet_src,
            alphabet_tar=self.alphabet_tar,
            num_h=self.num_h,
            h_degree=self.h_degree,
            use_dynamic_array_sizes=True)

        # load test set
        print('Load validation set')
        test_loader = tl.TextLoader(paths_X=self.test_x_files,
                                    paths_t=self.test_t_files,
                                    seq_len_x=self.seq_len_x,
                                    seq_len_t=self.seq_len_t)
        self.batch_generator['test'] = tl.TextBatchGenerator(
            loader=test_loader,
            batch_size=self.batch_size_valid,
            alphabet_src=self.alphabet_src,
            alphabet_tar=self.alphabet_tar,
            num_h=self.num_h,
            h_degree=self.h_degree,
            use_dynamic_array_sizes=True)


    def valid_dict(self, batch, feedback=True):
        """ Return feed_dict for validation """
        my_dict = { self.Xs:     batch['x_encoded'],
                    self.ts:     batch['t_encoded'],
                    self.ts_go:  batch['t_encoded_go'],
                    self.X_len:  batch['x_len'],
                    self.t_len:  batch['t_len'],
                    self.x_mask: batch['x_mask'],
                    self.t_mask: batch['t_mask'],
                    self.feedback: feedback }

        for i in range(self.num_h):
            my_dict[self.X_h[i]] = batch['x_h%d' % i]
            my_dict[self.X_h_len[i]] = batch['x_h%d_len' % i]

        return my_dict

    def train_dict(self, batch):
        """ Return feed_dict for training.
        Reuse validation feed_dict because the only difference is feedback.
        """
        return self.valid_dict(batch, feedback=False)

    def build_feed_dict(self, batch, validate=False):
        return self.valid_dict(batch) if validate else self.train_dict(batch)

    def get_generator(self, split):
        assert split in ['train', 'valid', 'test']
        return self.batch_generator[split].gen_batch

    def next_train_feed(self):
        generator = self.get_generator('train')
        for t_batch in generator(self.train_schedule_function):
            extra = { 't_len': t_batch['t_len'] }
            yield (self.build_feed_dict(t_batch), extra)

    def next_valid_feed(self):
        generator = self.get_generator('valid')
        for v_batch in generator(self.valid_schedule_function):
            yield self.build_feed_dict(v_batch, validate=True)

    def next_test_feed(self):
        generator = self.get_generator('test')
        for p_batch in generator(self.test_schedule_function):
            yield self.build_feed_dict(p_batch, validate=True)

    def get_alphabet_src(self):
        return self.batch_generator['train'].alphabet_src

    def get_alphabet_tar(self):
        return self.batch_generator['train'].alphabet_tar
