import time
import click
import os
import numpy as np
import tensorflow as tf
import importlib

import frostings.loader as frost
import text_loader as tl
from augmentor import Augmentor
from model import Model
from utils import basic as utils
import utils.performancemetrics as pm
from utils.tfhelper import run

SAVER_PATH = {'base': 'train/',
                     'checkpoint': 'checkpoints/',
                     'log': 'logs/'}


@click.command()
@click.option('--config', default='test', help='Config file to use for training')
class Trainer:
    """Train a translation model."""

    def __init__(self, config):
        self.setup_model(config)
        self.setup_reload_path()
        self.setup_loader()

        self.alphabet = self.batch_generator['train'].alphabet

        self.train()

    def setup_reload_path(self):
        self.latest_checkpoint = None
        self.checkpoint_saver = None
        self.summarywriter = None
        if not self.name:
            return  # Nothing more to do

        local_path        = os.path.join(SAVER_PATH['base'], self.name)
        self.summary_path = os.path.join(local_path, SAVER_PATH['log'])

        print("Will read and write from '%s' (checkpoints and logs)" % (local_path))

        # Prepare for saving checkpoints
        if self.save_freq:
            self.checkpoint_saver = tf.train.Saver()
            checkpoint_path = os.path.join(local_path, SAVER_PATH['checkpoint'])
            self.checkpoint_file_path = os.path.join(checkpoint_path, 'checkpoint')
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            self.latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        else:
            print("WARNING: 'save_freq' is 0 so checkpoints won't be saved!")

        # Prepare for writing TensorBoard summaries
        if self.tb_log_freq:
            if not os.path.exists(self.summary_path) and self.tb_log_freq:
                os.makedirs(self.summary_path)
            self.summarywriter = tf.train.SummaryWriter(self.summary_path)

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
        """Load the datasets"""
        # TODO: this will be moved to config files
        self.batch_generator = dict()

        # load training set
        print('\nload training set')
        train_loader = tl.TextLoader(
            paths_X=['data/train/europarl-v7.da-en.en'],
            paths_t=['data/train/europarl-v7.da-en.da'],
            seq_len=self.seq_len
        )
        train_iteration_schedule = frost.IterationSchedule(shuffle=True, repeat=True)
        self.batch_generator['train'] = tl.TextBatchGenerator(
            loader=train_loader,
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            iteration_schedule=train_iteration_schedule
        )

        # load validation set
        print('\nload validation set')
        valid_loader = tl.TextLoader(
            paths_X=['data/test/devtest2006.en', 'data/test/test2006.en'],
            paths_t=['data/test/devtest2006.da', 'data/test/test2006.da'],
            seq_len=self.seq_len
        )
        self.batch_generator['valid'] = tl.TextBatchGenerator(
            loader=valid_loader,
            batch_size=self.batch_size,
            seq_len=self.seq_len
        )

    def visualize_ys(self, ys, batch):
        for j in range(batch['x_encoded'].shape[0]):
            click.echo('%s ::: %s ::: %s' % (
                self.alphabet.decode(batch['x_encoded'][j]),
                self.alphabet.decode(ys[j]),
                self.alphabet.decode(batch['t_encoded'][j])
                ))

    def train(self):
        print("## INITIATE TRAIN")
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
            if self.latest_checkpoint:
                self.checkpoint_saver.restore(sess, self.latest_checkpoint)
            else:
                tf.initialize_all_variables().run()

            # prepare summary operations and summary writer
            summaries = tf.merge_all_summaries()
            self.val_summaries = self.setup_validation_summaries()

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

                fetches = {
                    'loss':      self.model.loss,
                    'ys':        self.model.ys,
                    'train_op':  self.model.train_op,
                    'accuracy':  self.model.accuracy,
                    'summaries': summaries
                }

                res, elapsed_it = self.perform_iteration(
                    sess,
                    fetches,
                    batch=t_batch,
                    feedback=self.train_feedback)

                combined_time += elapsed_it

                if self.visualize and i % self.visualize == 0:
                    self.visualize_ys(res['ys'], t_batch)

                if self.summarywriter and i % self.tb_log_freq == 0:
                    self.summarywriter.add_summary(res['summaries'], i)

                if self.checkpoint_saver and i and i % self.save_freq == 0:
                    self.checkpoint_saver.save(
                        sess,
                        self.checkpoint_file_path,
                        self.model.global_step
                    )

                if self.log_freq and i % self.log_freq == 0:
                    batch_acc = res['accuracy']
                    click.echo('Iteration %i\t Loss: %f\t Acc: %f\t Elapsed: %f (%f)' % (
                        i, np.mean(res['loss']), batch_acc, combined_time, (combined_time/self.log_freq) ))
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
        res = run(sess, fetches, feed_dict=feed_dict)
        elapsed = time.time() - start_time

        return (res, elapsed)

    def validate(self, sess):
        print("## VALIDATING")
        accuracies = []
        valid_ys = []
        valid_ts = []
        for v_batch in self.batch_generator['valid'].gen_batch():
            fetches = {'accuracy': self.model.accuracy,
                       'ys': self.model.ys}

            res, time = self.perform_iteration(
                sess,
                fetches,
                feed_dict=None,
                batch=v_batch,
                feedback=True)

            # TODO: accuracies should be weighted by batch sizes
            # before averaging
            valid_ys.append(res['ys'])
            valid_ts.append(v_batch['t_encoded'])
            accuracies.append(res['accuracy'])

        valid_ys = np.concatenate(valid_ys, axis=0)
        valid_ts = np.concatenate(valid_ts, axis=0)

        # print visualization
        self.visualize_ys(res['ys'], v_batch)

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

        if self.summarywriter:
            feed_dict = {
                self.model.accuracy: valid_acc,
                self.bleu: corpus_bleu,
                self.edit_dist: edit_dist
            }
            fetches = [self.val_summaries, self.model.global_step]
            summaries, i = sess.run(fetches, feed_dict)
            self.summarywriter.add_summary(summaries, i)

        print("\n## VALIDATION DONE")

if __name__ == '__main__':
    trainer = Trainer()
