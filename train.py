import time
import click
import numpy as np
import tensorflow as tf

import text_loader as tl

from frostings import loader as fl
from model import Model
from utils import acc, create_tsne as TSNE
from dummy_loader import DummySampleGenerator

USE_LOGGED_WEIGHTS = False
DEFAULT_VALIDATION_SPLIT = './data/validation_split_v1.pkl'

@click.command()
@click.option(
    '--loader', type=click.Choice(['europarl', 'normal', 'talord',
    'talord_caps1', 'talord_caps2', 'talord_caps3']), default='europarl',
    help='Choose dataset to load. (default: europarl)')
@click.option('--tsne', is_flag=True,
    help='Use t-sne to plot character embeddings.')
@click.option('--visualize', default=1000,
    help='Print visualizations every N iterations.')
@click.option('--log-freq', default=10,
    help='Print updates every N iterations. (default: 10)')
@click.option('--save-freq', default=0,
    help='Create checkpoint every N iterations.')
@click.option('--iterations', default=20000,
    help='Number of iterations (default: 20000)')
@click.option('--valid-freq', default=0,
    help='Validate every N iterations. 0 to disable. (default: 0)')
@click.option('--seq-len', default=50,
    help='Maximum length of char sequences')
class Trainer:
    """Train a translation model."""

    def __init__(self, loader, tsne, visualize, log_freq, save_freq, iterations, valid_freq, seq_len):
        self.loader, self.tsne, self.visualize = loader, tsne, visualize
        self.log_freq, self.save_freq, self.valid_freq = log_freq, save_freq, valid_freq
        self.iterations, self.seq_len = iterations, seq_len

        self.setup_placeholders()
        self.setup_model()
        self.setup_loader()
        self.setup_batch_generator()

        self.alphabet = self.batch_generator['train'].alphabet

        self.train()

    def setup_placeholders(self):
        self.Xs       = tf.placeholder(tf.int32,   shape=[None, self.seq_len], name='X_input')
        self.ts       = tf.placeholder(tf.int32,   shape=[None, self.seq_len], name='t_input')
        self.ts_go    = tf.placeholder(tf.int32,   shape=[None, self.seq_len], name='t_input_go')
        self.X_len    = tf.placeholder(tf.int32,   shape=[None],               name='X_len')
        self.t_mask   = tf.placeholder(tf.float32, shape=[None, self.seq_len], name='t_mask')
        self.feedback = tf.placeholder(tf.bool,                                name='feedback_indicator')

    def setup_model(self):
        self.model = Model(
                alphabet_size = 337,
                max_x_seq_len = self.seq_len,
                max_t_seq_len = self.seq_len )
        self.model.build(self.Xs, self.X_len, self.ts_go, self.feedback)
        self.model.build_loss(self.ts, self.t_mask)
        self.model.build_prediction()
        self.model.training(learning_rate = 0.1)

    def setup_loader(self):
        self.sample_generator = dict()
        if self.loader == 'europarl':
            print('Using europarl loader')
            self.load_method = {
                    'train': tl.TextLoadMethod(
                        ['data/train/europarl-v7.fr-en.en'],
                        ['data/train/europarl-v7.fr-en.fr'],
                        seq_len = self.seq_len ) }

            # TODO: making the validation split (should not just be True later on)
            # something like: `if not os.path.isfile(DEFAULT_VALIDATION_SPLIT):`
            if True:
                import create_validation_split as v_split
                no_training_samples = len(self.load_method['train'].samples)
                v_split.create_split(no_training_samples, DEFAULT_VALIDATION_SPLIT)

            split = np.load(DEFAULT_VALIDATION_SPLIT)
            self.sample_generator['train'] = tl.SampleTrainWrapper(
                    self.load_method['train'],
                    permutation = split['indices_train'],
                    num_splits = 32 )

            # data loader for eval
            # notice repeat = false
            self.eval_sample_generator = {
                'train':  fl.SampleGenerator(
                    self.load_method['train'],
                    permutation = split['indices_train'],
                    repeat = False ),
                'validation': fl.SampleGenerator(
                    self.load_method['train'], #TODO: is this the correct load method?
                    permutation = split['indices_valid'],
                    repeat = False ) }
        elif self.loader in [ 'normal', 'talord', 'talord_caps1', 'talord_caps2', 'talord_caps3']:
            print('Using dummy loader (%s)' % (self.loader) )
            self.sample_generator['train'] = DummySampleGenerator(
                    max_len = self.seq_len/6,
                    sampler = self.loader )
        else:
            raise NotImplementedError("Given loader (%s) is not supported" % (self.loader) )

    def setup_batch_generator(self):
        self.batch_generator = dict()
        if self.loader is 'europarl':
             self.batch_generator['train'] = tl.BatchTrainWrapper(
                    self.sample_generator['train'],
                    batch_size = 32,
                    seq_len = self.seq_len,
                    warm_up = 100 )
        else:
             self.batch_generator['train'] = tl.TextBatchGenerator(
                    self.sample_generator['train'],
                    batch_size = 32,
                    seq_len = self.seq_len )

        # If we have a validation frequency
        # setup needed evaluation generators
        if self.valid_freq:
            self.eval_batch_generator = {
                    # TODO: this was too large to run for now
                    #'train': tl.TextBatchGenerator(
                    #    self.eval_sample_generator['train'],
                    #    batch_size = 32,
                    #    seq_len = self.seq_len ),
                    'validation': tl.TextBatchGenerator(
                        self.eval_sample_generator['validation'],
                        batch_size = 32,
                        seq_len = self.seq_len ) }

    def train(self):
        print("## INITIATE TRAIN")
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # start from latest checkpoint
            if USE_LOGGED_WEIGHTS:
                latest_checkpoint = tf.train.latest_checkpoint('train/checkpoints')
                saver.restore(sess, latest_checkpoint)
            else:
                tf.initialize_all_variables().run()

            if self.tsne:
                TSNE(self.model, self.alphabet.decode_dict)

            summaries = tf.merge_all_summaries()
            writer = tf.train.SummaryWriter("train/logs", sess.graph_def)

            print("## TRAINING...")
            combined_time = 0.0 # total time for each print
            for i, t_batch in enumerate(self.batch_generator['train'].gen_batch()):
                ## VALIDATION START ##
                if self.valid_freq and i % self.valid_freq == 0:
                    print("## VALIDATING")
                    accuracies = []
                    for v_batch in self.eval_batch_generator['validation'].gen_batch():
                        # running the model only for inference
                        fetches = [ self.model.accuracy ]

                        res, elapsed_it = self.perform_iteration(sess, fetches, None, v_batch, True)

                        # TODO: accuracies should be weighted by batch sizes
                        # before averaging
                        accuracies.append(res[0])
                    # getting validation
                    valid_acc = np.mean(accuracies)
                    print('accuray:\t%.2f%% \n' % (valid_acc * 100))
                    print("## VALIDATION DONE")
                ## VALIDATION END ##

                ## TRAIN START ##
                fetches = [
                        self.model.loss,
                        self.model.ys,
                        summaries,
                        self.model.train_op,
                        self.model.accuracy ]

                res, elapsed_it = self.perform_iteration(sess, fetches, None, t_batch)
                ## TRAIN END ##

                combined_time += elapsed_it

                # Maybe visualize predictions
                if self.visualize and i % self.visualize == 0:
                    for j in range(32):
                        click.echo('%s ::: %s ::: %s' % (
                                self.alphabet.decode(t_batch['x_encoded'][j]),
                                self.alphabet.decode(res[1][j]),
                                self.alphabet.decode(t_batch['t_encoded'][j])
                            ))

                # Write summaries for TensorBoard.
                writer.add_summary(res[2], i)

                if self.save_freq and i and i % self.save_freq == 0:
                    saver.save(sess,
                               'train/checkpoints/checkpoint',
                               global_step = self.model.global_step)

                if self.log_freq and i % self.log_freq == 0:
                    batch_acc = res[4]
                    click.echo('Iteration %i\t Loss: %f\t Acc: %f\t Elapsed: %f (%f)' % (
                        i, np.mean(res[0]), batch_acc, combined_time, (combined_time/self.log_freq) ))
                    combined_time = 0.0

                if i >= self.iterations:
                    click.echo('reached max iteration: %d' % i)
                    break

    def perform_iteration(self, sess, fetches, feed_dict=None, batch=None, feedback=False):
        """ Performs one iteration/run.
            Returns tuple containing result and elapsed iteration time.

            Keyword arguments:
            sess:       Tensorflow Session
            fetches:    A single graph element, or a list of graph elements
            feed_dict:  A dictionary that maps graph elements to values (default: None)
            batch:      A batch with data used to fill feed_dict (default: None)
            feedback:   If true the decoder will get its own prediction the previous time step.
                        If false it will get target for previous time step (default: False)
        """
        if not fetches:
            raise ValueError("fetches argument must be a non-empty list")
        if type(feedback) is not bool:
            raise ValueError("feedback argument must be a boolean")

        if feed_dict is None and batch is not None:
            feed_dict = {
                    self.Xs:     batch['x_encoded'],
                    self.ts:     batch['t_encoded'],
                    self.ts_go:  batch['t_encoded_go'],
                    self.X_len:  batch['x_len'],
                    self.t_mask: batch['t_mask'],
                    self.feedback: feedback }

        start_time = time.time()
        res = sess.run(fetches, feed_dict=feed_dict)
        elapsed = time.time() - start_time

        return (res, elapsed)

if __name__ == '__main__':
    trainer = Trainer()
