import time
import click
import os
import numpy as np
import tensorflow as tf
from warnings import warn
import importlib

import text_loader as tl

from augmentor import Augmentor
from frostings import loader as fl
from model import Model
from utils import basic as utils
from dummy_loader import DummySampleGenerator
import utils.performancemetrics as pm

SAVER_FOLDER_PATH = {'base': 'train/',
                     'checkpoint': 'checkpoints/',
                     'log': 'logs/'}
USE_LOGGED_WEIGHTS = False
DEFAULT_VALIDATION_SPLIT = './data/validation_split_v1.pkl'


@click.command()
@click.option(
    '--loader',
    type=click.Choice(['europarl',
                       'normal',
                       'talord',
                       'talord_caps1',
                       'talord_caps2',
                       'talord_caps3']),
    default='europarl',
    help='Choose dataset to load. (default: europarl)')
@click.option('--config-name', default='test',
    help='Configuration file to use for model')
class Trainer:
    """Train a translation model."""

    def __init__(self, loader, config_name):
        self.loader = loader

        self.setup_model(config_name)
        self.setup_reload_path()
        self.setup_loader()
        self.setup_batch_generator()

        self.alphabet = self.batch_generator['train'].alphabet

        self.train()

    def setup_reload_path(self):
        self.named_checkpoint_path = self.named_log_path = self.checkpoint_file_path = None
        if self.name:
            USE_LOGGED_WEIGHTS = True

            local_folder_path           = os.path.join(SAVER_FOLDER_PATH['base'], self.name)
            self.named_checkpoint_path  = os.path.join(local_folder_path, SAVER_FOLDER_PATH['checkpoint'])
            self.named_log_path         = os.path.join(local_folder_path, SAVER_FOLDER_PATH['log'])

            self.checkpoint_file_path   = os.path.join(self.named_checkpoint_path, 'checkpoint')

            # make sure checkpoint folder exists
            if not os.path.exists(self.named_checkpoint_path):
                os.makedirs(self.named_checkpoint_path)
            if not os.path.exists(self.named_log_path):
                os.makedirs(self.named_log_path)

            print("Will read and write from '%s' (checkpoints and logs)" % (local_folder_path))
            if not self.save_freq:
                warn("'save_freq' is 0, won't save checkpoints", UserWarning)

    def setup_placeholders(self, seq_len):
        self.Xs       = tf.placeholder(tf.int32,   shape=[None, seq_len], name='X_input')
        self.ts       = tf.placeholder(tf.int32,   shape=[None, seq_len], name='t_input')
        self.ts_go    = tf.placeholder(tf.int32,   shape=[None, seq_len], name='t_input_go')
        self.X_len    = tf.placeholder(tf.int32,   shape=[None],          name='X_len')
        self.X_spaces = tf.placeholder(tf.int32,   shape=[None, seq_len//4], name='X_spaces')
        self.X_spaces_len = tf.placeholder(tf.int32, shape=[None],        name='X_spaces_len')
        self.t_mask   = tf.placeholder(tf.float32, shape=[None, seq_len], name='t_mask')
        self.feedback = tf.placeholder(tf.bool,                           name='feedback_indicator')

    def setup_validation_summaries(self):
        """A hack for recording performance metrics with TensorBoard."""
        self.bleu = tf.placeholder(tf.float32)
        self.edit_dist = tf.placeholder(tf.float32)

        valid_summaries = [
            tf.scalar_summary('validation/accuracy', self.model.accuracy),
            tf.scalar_summary('validation/bleu', self.bleu),
            tf.scalar_summary('validation/edit dist per char', self.edit_dist)
        ]

        return tf.merge_summary(valid_summaries)

    def setup_model(self, config_name):
        # load the config module to use
        config_path = 'configurations.' + config_name
        config = importlib.import_module(config_path)

        # copy settings that affect the training script
        self.batch_size = config.Model.batch_size
        self.seq_len = config.Model.seq_len
        self.name = config.Model.name
        self.visualize = config.Model.visualize_freq
        self.log_freq = config.Model.log_freq
        self.save_freq = config.Model.save_freq
        self.valid_freq = config.Model.valid_freq
        self.iterations = config.Model.iterations
        self.warm_up = config.Model.warmup
        self.train_feedback = config.Model.train_feedback
        self.tb_log_freq = config.Model.tb_log_freq

        # Create placeholders and construct model
        self.setup_placeholders(config.Model.seq_len)
        self.model = config.Model(
                Xs=self.Xs,
                X_len=self.X_len,
                ts=self.ts,
                ts_go=self.ts_go,
                t_mask=self.t_mask,
                feedback=self.feedback,
                X_spaces=self.X_spaces,
                X_spaces_len=self.X_spaces_len)

    def setup_loader(self):
        self.sample_generator = dict()
        if self.loader == 'europarl':
            print('Using europarl loader')
            lm = tl.TextLoadMethod(
                paths_X=['data/train/europarl-v7.da-en.en'],
                paths_t=['data/train/europarl-v7.da-en.da'],
                seq_len=self.seq_len)
            self.load_method = {'train': lm}

            # TODO: making the validation split (should not just be True later
            # on) something like: `if not
            # os.path.isfile(DEFAULT_VALIDATION_SPLIT):`
            if True:
                import create_validation_split as v_split
                no_training_samples = len(self.load_method['train'].samples)
                v_split.create_split(no_training_samples,
                                     DEFAULT_VALIDATION_SPLIT)

            split = np.load(DEFAULT_VALIDATION_SPLIT)
            self.sample_generator['train'] = tl.SampleTrainWrapper(
                    self.load_method['train'],
                    permutation=split['indices_train'],
                    num_splits=32)

            # data loader for eval
            # notice repeat = false
            self.eval_sample_generator = {
                'train': fl.SampleGenerator(
                    self.load_method['train'],
                    permutation=split['indices_train'],
                    repeat=False),
                'validation': fl.SampleGenerator(
                    self.load_method['train'],  # TODO: is this the correct load method?
                    permutation=split['indices_valid'],
                    repeat=False)}
        elif self.loader in ['normal', 'talord', 'talord_caps1', 'talord_caps2', 'talord_caps3']:
            print('Using dummy loader (%s)' % (self.loader))
            self.sample_generator['train'] = DummySampleGenerator(
                    max_len=self.seq_len/6,
                    sampler=self.loader)
        else:
            raise NotImplementedError("Given loader (%s) is not supported" % (self.loader))

    def setup_batch_generator(self):
        self.batch_generator = dict()
        if self.loader is 'europarl':
            self.batch_generator['train'] = tl.BatchTrainWrapper(
                self.sample_generator['train'],
                batch_size=self.batch_size,
                seq_len=self.seq_len,
                warm_up=self.warm_up)
        else:
            self.batch_generator['train'] = tl.TextBatchGenerator(
                self.sample_generator['train'],
                batch_size=self.batch_size,
                seq_len=self.seq_len)

        # If we have a validation frequency
        # setup needed evaluation generators
        if self.valid_freq:
            self.eval_batch_generator = {
                    # TODO: this was too large to run for now
                    # 'train': tl.TextBatchGenerator(
                    #     self.eval_sample_generator['train'],
                    #     batch_size = self.batch_size,
                    #     seq_len = self.seq_len ),
                    'validation': tl.TextBatchGenerator(
                        self.eval_sample_generator['validation'],
                        batch_size=self.batch_size,
                        seq_len=self.seq_len)}

    def visualize_ys(self, ys, batch):
        for j in range(batch['x_encoded'].shape[0]):
            click.echo('%s ::: %s ::: %s' % (
                self.alphabet.decode(batch['x_encoded'][j]),
                self.alphabet.decode(ys[j]),
                self.alphabet.decode(batch['t_encoded'][j])
                ))

    def train(self):
        print("## INITIATE TRAIN")

        with tf.Session() as sess:
            saver = tf.train.Saver()
            # restore only if files exist
            if USE_LOGGED_WEIGHTS and os.path.exists(self.named_checkpoint_path) and os.listdir(self.named_checkpoint_path):
                latest_checkpoint = tf.train.latest_checkpoint(self.named_checkpoint_path)
                saver.restore(sess, latest_checkpoint)
            else:
                tf.initialize_all_variables().run()

            # prepare summary operations and summary writer
            summaries = tf.merge_all_summaries()
            self.val_summaries = self.setup_validation_summaries()
            if self.named_log_path and os.path.exists(self.named_log_path):
                writer = tf.train.SummaryWriter(self.named_log_path, sess.graph_def)
                self.writer = writer

            print("## TRAINING...")
            combined_time = 0.0  # total time for each print
            swap_amount = None
            augmentor = Augmentor()
            for i, t_batch in enumerate(self.batch_generator['train'].gen_batch()):

                if i in self.model.swap_schedule:
                    swap_amount = self.model.swap_schedule[i]
                    print(" setting swap amount to %0.4f" % swap_amount)
                if swap_amount > 0.0:
                    t_batch['t_encoded_go'] = augmentor.run(
                        t_batch['t_encoded_go'], t_batch['t_len'], swap_amount,
                        1)

                if self.valid_freq and i % self.valid_freq == 0:
                    self.validate(sess)

                ## TRAIN START ##
                fetches = [
                        self.model.loss,
                        self.model.ys,
                        summaries,
                        self.model.train_op,
                        self.model.accuracy ]

                res, elapsed_it = self.perform_iteration(
                    sess,
                    fetches,
                    batch=t_batch,
                    feedback=self.train_feedback)
                ## TRAIN END ##

                combined_time += elapsed_it

                if self.visualize and i % self.visualize == 0:
                    self.visualize_ys(res[1], t_batch)

                if self.named_log_path and os.path.exists(self.named_log_path) and i % self.tb_log_freq == 0:
                    writer.add_summary(res[2], i)

                if self.save_freq and i and i % self.save_freq == 0 and self.named_checkpoint_path:
                    saver.save(sess, self.checkpoint_file_path, self.model.global_step)

                if self.log_freq and i % self.log_freq == 0:
                    batch_acc = res[4]
                    click.echo('Iteration %i\t Loss: %f\t Acc: %f\t Elapsed: %f (%f)' % (
                        i, np.mean(res[0]), batch_acc, combined_time, (combined_time/self.log_freq) ))
                    combined_time = 0.0

                if i >= self.iterations:
                    click.echo('reached max iteration: %d' % i)
                    break

    def perform_iteration(self, sess, fetches, feed_dict=None, batch=None,
                          feedback=False):
        """ Performs one iteration/run.

            Returns tuple containing result and elapsed iteration time.

            Keyword arguments:
            sess:       Tensorflow Session
            fetches:    A single graph element, or a list of graph
                        elements.
            feed_dict:  A dictionary that maps graph elements to values
                        (default: None)
            batch:      A batch with data used to fill feed_dict
                        (default: None)
            feedback:   If true the decoder will get its own prediction
                        the previous time step. If false it will get
                        target for previous time step (default: False)
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
                    self.feedback: feedback,
                    self.X_spaces: batch['x_spaces'],
                    self.X_spaces_len: batch['x_spaces_len']}

        start_time = time.time()
        res = sess.run(fetches, feed_dict=feed_dict)
        elapsed = time.time() - start_time

        return (res, elapsed)

    def validate(self, sess):
        print("## VALIDATING")
        accuracies = []
        valid_ys = []
        valid_ts = []
        for v_batch in self.eval_batch_generator['validation'].gen_batch():
            fetches = [self.model.accuracy, self.model.ys]

            res, time = self.perform_iteration(
                sess,
                fetches,
                feed_dict=None,
                batch=v_batch,
                feedback=True)

            # TODO: accuracies should be weighted by batch sizes
            # before averaging
            valid_ys.append(res[1])
            valid_ts.append(v_batch['t_encoded'])
            accuracies.append(res[0])

        valid_ys = np.concatenate(valid_ys, axis=0)
        valid_ts = np.concatenate(valid_ts, axis=0)

        # print visualization
        self.visualize_ys(res[1], v_batch)

        # convert all predictions to strings
        str_ts, str_ys = utils.numpy_to_words(valid_ts,
                                              valid_ys,
                                              self.alphabet)

        # accuracy
        valid_acc = np.mean(accuracies)
        print('\t%s%.2f%%' % ('accuracy:'.ljust(25), (valid_acc * 100)))

        # BLEU score
        corpus_bleu = pm.corpus_bleu(str_ys, str_ts)
        print('\t%s%.5f' % ('BLEU:'.ljust(25), corpus_bleu))

        # edit distance
        edit_dist = pm.mean_char_edit_distance(str_ys, str_ts)
        print('\t%s%f' % ('Mean edit dist per char:'.ljust(25), edit_dist))

        if self.named_log_path and os.path.exists(self.named_log_path):
            feed_dict = {
                self.model.accuracy: valid_acc,
                self.bleu: corpus_bleu,
                self.edit_dist: edit_dist
            }
            fetches = [self.val_summaries, self.model.global_step]
            summaries, i = sess.run(fetches, feed_dict)
            self.writer.add_summary(summaries, i)

        print("\n## VALIDATION DONE")

if __name__ == '__main__':
    trainer = Trainer()
