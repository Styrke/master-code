import tensorflow as tf
import text_loader as tl
from utils.tfextensions import sequence_loss_tensor
from utils.tfextensions import _grid_gather
from utils.tfextensions import mask
from utils.rnn import encoder
from utils.rnn import attention_decoder
from data.alphabet import Alphabet

class Model:
    num_gpus = 1
    # settings that affect train.py
    batch_size = 512
    seq_len = 50
    name = None  # (string) For saving logs and checkpoints. (None to disable.)
    visualize_freq = 100000  # Visualize training X, y, and t. (0 to disable.)
    log_freq = 100  # How often to print updates during training.
    save_freq = 20000  # How often to save checkpoints. (0 to disable.)
    valid_freq = 500  # How often to validate.
    iterations = 5*32000  # How many iterations to train for before stopping.
    train_feedback = False  # Enable feedback during training?
    tb_log_freq = 500  # How often to save logs for TensorBoard

    # datasets
    train_x_files = ['data/train/europarl-v7.fr-en.en']
    train_t_files = ['data/train/europarl-v7.fr-en.fr']
    valid_x_files = ['data/valid/devtest2006.en', 'data/valid/test2006.en',
                     'data/valid/test2007.en', 'data/valid/test2008.en']
    valid_t_files = ['data/valid/devtest2006.fr', 'data/valid/test2006.fr',
                     'data/valid/test2007.fr', 'data/valid/test2008.fr']

    # settings that are local to the model
    alphabet_src_size = 302  # size of alphabet
    alphabet_tar_size = 303  # size of alphabet
    alphabet_src = Alphabet('data/alphabet/dict.fr-en.en', eos='*') 
    alphabet_tar = Alphabet('data/alphabet/dict.fr-en.fr', eos='*', sos='')
    char_encoder_units = 400  # number of units in character-level encoder
    word_encoder_units = 400  # num nuits in word-level encoders (both forwards and back)
    embedd_dims = 256  # size of character embeddings
    learning_rate = 0.001
    reg_scale = 0.000001
    clip_norm = 1

    swap_schedule = {
        0: 0.0
    }

    # kwargs for scheduling function
    schedule_kwargs = {
        'warmup_iterations': 0,  # if warmup_schedule is used
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
        #self.placeholders = self.setup_placeholders()

        # schedule functions
        self.train_schedule_function = tl.warmup_schedule
        self.valid_schedule_function = None  # falls back to frostings.default_schedule

        print("Model instantiation")
        #out_tensor, valid_out_tensor = self.build(self.placeholders)

        self.placeholders, self.out, self.out_valid, self.loss, self.loss_valid, self.acc, self.acc_valid, self.train_op = self.build_towers()
        #self.loss, self.accuracy = self.build_loss(out_tensor, self.placeholders)
        #self.valid_loss, self.valid_accuracy = self.build_valid_loss(valid_out_tensor, self.placeholders)
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            # do this on the CPU (for now ..!)
            self.ys = self.build_prediction(self.out)
            self.valid_ys = self.build_valid_prediction(self.out_valid)
        #self.build_training(self.loss)

        # Create TensorBoard scalar summaries
        tf.scalar_summary('train/loss', self.loss)
        tf.scalar_summary('train/accuracy', self.acc)

        # setup batch generators
        self.setup_batch_generators()

    def setup_placeholders(self, scope):
        ph_dict = dict()
        shape = [None, None]
        ph_dict['Xs']       = tf.placeholder(tf.int32, shape=shape, name='X_input')
        ph_dict['ts']       = tf.placeholder(tf.int32, shape=shape, name='t_input')
        ph_dict['ts_go']    = tf.placeholder(tf.int32, shape=shape, name='t_input_go')
        ph_dict['X_len']    = tf.placeholder(tf.int32, shape=[None], name='X_len')
        ph_dict['t_len']    = tf.placeholder(tf.int32, shape=[None], name='t_len')
        ph_dict['feedback'] = tf.placeholder(tf.bool, name='feedback_indicator')
        ph_dict['X_mask']   = tf.placeholder(tf.float32, shape=shape, name='X_mask')
        ph_dict['t_mask']   = tf.placeholder(tf.float32, shape=shape, name='t_mask')

        shape = [None, None]
        ph_dict['X_spaces']     = tf.placeholder(tf.int32, shape=shape,  name='X_spaces')
        ph_dict['X_spaces_len'] = tf.placeholder(tf.int32, shape=[None], name='X_spaces_len')
        return ph_dict

    def build_inference(self, placeholders):
        print('Building model')
        x_embeddings = tf.Variable(
            tf.random_normal([self.alphabet_src_size, self.embedd_dims],
            stddev=0.1), name='x_embeddings')
        t_embeddings = tf.Variable(
            tf.random_normal([self.alphabet_tar_size, self.embedd_dims],
            stddev=0.1), name='t_embeddings')

        X_embedded = tf.gather(x_embeddings, placeholders['Xs'], name='embed_X')
        t_embedded = tf.gather(t_embeddings, placeholders['ts_go'], name='embed_t')

        with tf.variable_scope('dense_out'):
            W_out = tf.get_variable('W_out', [self.word_encoder_units*2, self.alphabet_tar_size])
            b_out = tf.get_variable('b_out', [self.alphabet_tar_size])

        # forward encoding
        char_enc_state, char_enc_out = encoder(X_embedded, placeholders['X_len'], 'char_encoder', self.char_encoder_units)
        char2word = _grid_gather(char_enc_out, placeholders['X_spaces'])
        char2word.set_shape([None, None, self.char_encoder_units])
        word_enc_state, word_enc_out = encoder(char2word, placeholders['X_spaces_len'], 'word_encoder', self.word_encoder_units)

        # backward encoding words
        char2word = tf.reverse_sequence(char2word, tf.to_int64(placeholders['X_spaces_len']), 1)
        char2word.set_shape([None, None, self.char_encoder_units])
        word_enc_state_bck, word_enc_out_bck = encoder(char2word, placeholders['X_spaces_len'], 'word_encoder_backwards', self.word_encoder_units)
        word_enc_out_bck = tf.reverse_sequence(word_enc_out_bck, tf.to_int64(placeholders['X_spaces_len']), 1)

        word_enc_state = tf.concat(1, [word_enc_state, word_enc_state_bck])
        word_enc_out = tf.concat(2, [word_enc_out, word_enc_out_bck])

        # decoding
        dec_state, dec_out, valid_dec_out = (
            attention_decoder(word_enc_out, placeholders['X_spaces_len'], word_enc_state,
                              t_embedded, placeholders['t_len'], self.word_encoder_units,
                              t_embeddings, W_out, b_out))

        out_tensor = tf.reshape(dec_out, [-1, self.word_encoder_units*2])
        out_tensor = tf.matmul(out_tensor, W_out) + b_out
        out_shape = tf.concat(0, [tf.expand_dims(tf.shape(placeholders['X_len'])[0], 0),
                                  tf.expand_dims(tf.shape(t_embedded)[1], 0),
                                  tf.expand_dims(tf.constant(self.alphabet_tar_size), 0)])
        out_tensor = tf.reshape(out_tensor, out_shape)
        out_tensor.set_shape([None, None, self.alphabet_tar_size])

        valid_out_tensor = tf.reshape(valid_dec_out, [-1, self.word_encoder_units*2])
        valid_out_tensor = tf.matmul(valid_out_tensor, W_out) + b_out
        valid_out_tensor = tf.reshape(valid_out_tensor, out_shape)

        # add TensorBoard summaries for all variables
        tf.contrib.layers.summarize_variables()
        return out_tensor, valid_out_tensor

    def build_loss(self, out_tensor, placeholders):
        """Build a loss function and accuracy for the model."""
        print('  Building loss and accuracy')

        with tf.variable_scope('accuracy'):
            argmax = tf.to_int32(tf.argmax(out_tensor, 2))
            correct = tf.to_float(tf.equal(argmax, placeholders['ts'])) * placeholders['t_mask']
            accuracy = tf.reduce_sum(correct) / tf.reduce_sum(placeholders['t_mask'])

        with tf.variable_scope('loss'):
            loss = sequence_loss_tensor(out_tensor, placeholders['ts'], placeholders['t_mask'],
                    self.alphabet_tar_size)

            with tf.variable_scope('regularization'):
                regularize = tf.contrib.layers.l2_regularizer(self.reg_scale)
                params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                reg_term = sum([regularize(param) for param in params])

            loss += reg_term

        return loss, accuracy

    def build_valid_loss(self, valid_out_tensor, placeholders):
        return self.build_loss(valid_out_tensor, placeholders)

    def build_prediction(self, out_tensor):
        print('  Building prediction')
        with tf.variable_scope('prediction'):
            # logits is a list of tensors of shape [batch_size, alphabet_size].
            # We need shape of [batch_size, target_seq_len, alphabet_size].
            return tf.argmax(out_tensor, dimension=2)

    def build_valid_prediction(self, valid_out_tensor):
        return self.build_prediction(valid_out_tensor)

    def build_training(self, scope, loss, optimizer):
        print('  Building training')

        # Do gradient clipping
        # NOTE: this is the correct, but slower clipping by global norm.
        # Maybe it's worth trying the faster tf.clip_by_norm()
        # (See the documentation for tf.clip_by_global_norm() for more info)
        grads_and_vars = optimizer.compute_gradients(loss)
        #gradients, variables = zip(*grads_and_vars)  # unzip list of tuples
        #clipped_gradients, global_norm = (
        #        tf.clip_by_global_norm(gradients, self.clip_norm) )
        #clipped_grads_and_vars = zip(clipped_gradients, variables)

        # Create TensorBoard scalar summary for global gradient norm
        #tf.scalar_summary('train/global gradient norm', global_norm)

        # Create TensorBoard summaries for gradients
        # for grad, var in grads_and_vars:
        #     # Sparse tensor updates can't be summarized, so avoid doing that:
        #     if isinstance(grad, tf.Tensor):
        #         tf.histogram_summary('grad_' + var.name, grad)

        # make training op for applying the gradients
        #self.train_op = optimizer.apply_gradients(clipped_grads_and_vars,
        #                                          global_step=self.global_step)
        return grads_and_vars#clipped_grads_and_vars

    def build_towers(self):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            # connecting inference + loss for each tower.
            def build_tower(scope, optimizer):
                tower = dict()
                placeholders = self.setup_placeholders(scope)
                tower['placeholders'] = placeholders
                out, out_valid = self.build_inference(placeholders)
                tower['out'] = out
                tower['out_valid'] = out_valid
                loss, acc = self.build_loss(out, placeholders)
                tower['loss'] = loss
                tower['acc'] = acc
                loss_valid, acc_valid = self.build_valid_loss(out_valid, placeholders)
                tower['loss_valid'] = loss_valid
                tower['acc_valid'] = acc_valid
                return tower

            def average_gradients(tower_grads):
                """Calculate the average gradient for each shared variable across all towers.
                Note that this function provides a synchronization point across all towers.
                Args:
                    tower_grads: List of lists of (gradient, variable) tuples. The outer list
                        is over individual gradients. The inner list is over the gradient
                        calculation for each tower.
                Returns:
                    List of pairs of (gradient, variable) where the gradient has been averaged
                    across all towers.
                """
                average_grads = []
                for grad_and_vars in zip(*tower_grads):
                    # Note that each grad_and_vars looks like the following:
                    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
                    grads = []
                    for g, _ in grad_and_vars:
                        # Add 0 dimension to the gradients to represent the tower.
                        expanded_g = tf.expand_dims(g, 0)

                        # Append on a 'tower' dimension which we will average over below.
                        grads.append(expanded_g)

                    # Average over the 'tower' dimension.
                    grad = tf.concat(0, grads)
                    grad = tf.reduce_mean(grad, 0)

                    # Keep in mind that the Variables are redundant because they are shared
                    # across towers. So .. we will just return the first tower's pointer to
                    # the Variable.
                    v = grad_and_vars[0][1]
                    grad_and_var = (grad, v)
                    average_grads.append(grad_and_var)
                return average_grads

            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

            placeholders = []
            tower_out = []
            tower_out_valid = []
            tower_loss = []
            tower_loss_valid = []
            tower_acc = []
            tower_acc_valid = []
            tower_grads = []
            for i in range(self.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        tower = build_tower(scope, optimizer)
                        tf.get_variable_scope().reuse_variables()
                        grads = self.build_training(scope, tower['loss'], optimizer)
                        placeholders.append(tower['placeholders'])
                        tower_out.append(tower['out'])
                        tower_out_valid.append(tower['out_valid'])
                        tower_loss.append(tower['loss'])
                        tower_loss_valid.append(tower['loss_valid'])
                        tower_acc.append(tower['acc'])
                        tower_acc_valid.append(tower['acc_valid'])
                        tower_grads.append(grads)

            out = tf.concat(0, tower_out)
            out_valid = tf.concat(0, tower_out_valid)
            loss = tf.add_n(tower_loss)
            loss_valid = tf.add_n(tower_loss_valid)/self.num_gpus
            acc = tf.add_n(tower_acc)
            acc_valid = tf.add_n(tower_acc_valid)             

            grads = average_gradients(tower_grads)

            train_op = optimizer.apply_gradients(grads,
                                                 global_step=self.global_step)
        return placeholders, out, out_valid, loss, loss_valid, acc, acc_valid, train_op

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
            alphabet_src=self.alphabet_src,
            alphabet_tar=self.alphabet_tar,
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
            alphabet_src=self.alphabet_src,
            alphabet_tar=self.alphabet_tar,
            use_dynamic_array_sizes=True)

    def valid_dict(self, batch, placeholders, feedback=True):
        """ Return feed_dict for validation """
        return { placeholders['Xs']:           batch['x_encoded'],
                 placeholders['ts']:           batch['t_encoded'],
                 placeholders['ts_go']:        batch['t_encoded_go'],
                 placeholders['X_len']:        batch['x_len'],
                 placeholders['t_len']:        batch['t_len'],
                 placeholders['X_mask']:       batch['x_mask'],
                 placeholders['t_mask']:       batch['t_mask'],
                 placeholders['feedback']:     feedback,
                 placeholders['X_spaces']:     batch['x_spaces'],
                 placeholders['X_spaces_len']: batch['x_spaces_len'] }

    def train_dict(self, batch, placeholders):
        """ Return feed_dict for training.
        Reuse validation feed_dict because the only difference is feedback.
        """
        return self.valid_dict(batch, placeholders, feedback=False)

    def build_feed_dict(self, batch, placeholders, validate=False):
        return self.valid_dict(batch, placeholders) if validate else self.train_dict(batch, placeholders)

    def get_generator(self, validate=False):
        k = 'valid' if validate else 'train'
        return self.batch_generator[k].gen_batch

    def next_train_feed(self, placeholders):
        generator = self.get_generator()
        for t_batch in generator(self.train_schedule_function):
            #for g in gpu_towers:
            #    placeholders = g['placeholder']
            #    henning.append(self.build_feed_dict(t_batch, placeholders), extra)
            placeholders = self.placeholders
            extra = { 't_len': t_batch['t_len'] }
            yield (self.build_feed_dict(t_batch, placeholders), extra)

    def next_valid_feed(self):
        generator = self.get_generator(validate=True)
        for v_batch in generator(self.valid_schedule_function):
            placeholders = self.placeholders
            yield self.build_feed_dict(v_batch, placeholders, validate=True)

    def get_alphabet_src(self):
        return self.batch_generator['train'].alphabet_src

    def get_alphabet_tar(self):
        return self.batch_generator['train'].alphabet_tar
