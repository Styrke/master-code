import time
import click
import os
import numpy as np
import tensorflow as tf
import importlib

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
        self.load_config(config)
        self.setup_reload_path()

        self.alphabet = self.model.get_alphabet()

        self.train()

    def setup_reload_path(self):
        self.latest_checkpoint = None
        self.checkpoint_saver = None
        self.summarywriter = None
        if not self.name:
            return  # Nothing more to do

        local_path        = os.path.join(SAVER_PATH['base'], self.name)
        self.summary_path = os.path.join(local_path, SAVER_PATH['log'])

        print("Will read and write from '{:s}' (checkpoints and logs)".format(local_path))

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

    def load_config(self, config_name):
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
        self.train_feedback = config.Model.train_feedback
        self.tb_log_freq = config.Model.tb_log_freq

        # self.config is just an alias for the model for now. This may change later
        self.config = self.model = config.Model()

    def visualize_ys(self, ys, feed_dict):
        sep = ":::"
        pred_len = len(max(ys, key=len)) # length of longest predicted string
        for j in range(feed_dict[self.model.Xs].shape[0]):
            inp  = self.alphabet.decode(feed_dict[self.model.Xs][j]).ljust(self.seq_len)
            pred = self.alphabet.decode(ys[j]).ljust(pred_len)
            targ = self.alphabet.decode(feed_dict[self.model.ts][j])
            print('{1} {0} {2} {0} {3}'.format(sep, inp, pred, targ))

    def train(self):
        print("Training..")
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
            if self.latest_checkpoint:
                self.checkpoint_saver.restore(sess, self.latest_checkpoint)
            else:
                tf.initialize_all_variables().run()

            # prepare summary operations and summary writer
            summaries = tf.merge_all_summaries()
            self.val_summaries = self.setup_validation_summaries()

            combined_time = 0.0  # total time for each print
            swap_amount = None
            augmentor = Augmentor()
            for t_feed_dict, extra in self.model.next_train_feed():
                # NOTE is this slower than enumerate()?
                i = self.model.global_step.eval()

                if i in self.model.swap_schedule:
                    swap_amount = self.model.swap_schedule[i]
                    print("  setting swap amount to {:.4f}".format(swap_amount))
                if swap_amount > 0.0:
                    t_feed_dict[self.model.ts] = augmentor.run(
                            t_feed_dict[self.model.ts], extra['t_len'],
                            swap_amount, skip_left=1)

                if self.valid_freq and i % self.valid_freq == 0:
                    self.validate(sess)

                fetches = { 'loss':      self.model.loss,
                            'ys':        self.model.ys,
                            'train_op':  self.model.train_op,
                            'accuracy':  self.model.accuracy,
                            'summaries': summaries }

                res, elapsed_it = self.perform_iteration(sess, fetches,
                        t_feed_dict)

                combined_time += elapsed_it

                if self.visualize and i % self.visualize == 0:
                    self.visualize_ys(res['ys'], t_feed_dict)

                if self.summarywriter and i % self.tb_log_freq == 0:
                    self.summarywriter.add_summary(res['summaries'], i)

                if self.checkpoint_saver and i and i % self.save_freq == 0:
                    self.checkpoint_saver.save(sess, self.checkpoint_file_path,
                            self.model.global_step)

                if self.log_freq and i % self.log_freq == 0:
                    out = "Iteration {:d}\tLoss {:f}\tAcc: {:f}\tElapsed: {:f} ({:f})"
                    print(out.format(i, np.mean(res['loss']), res['accuracy'],
                        combined_time, (combined_time/self.log_freq)))
                    combined_time = 0.0

                if i >= self.iterations:
                    print('Reached max iteration: {:d}'.format(i))
                    break

    def perform_iteration(self, sess, fetches, feed_dict):
        """ Performs one iteration/run.

            Returns tuple containing result and elapsed iteration time.

            Keyword arguments:
            sess      -- Tensorflow Session
            fetches   -- A iterable of graph elements
            feed_dict -- A dictionary that maps graph elements to values
        """
        if not fetches:
            raise ValueError("fetches argument must be a non-empty list")

        start_time = time.time()
        res = run(sess, fetches, feed_dict=feed_dict)
        elapsed = time.time() - start_time

        return (res, elapsed)

    def validate(self, sess):
        print("Validating..")
        total_num_samples = 0
        accuracies, valid_ys, valid_ts = [], [], []
        for v_feed_dict in self.model.next_valid_feed():
            fetches = {'accuracy': self.model.accuracy,
                       'ys': self.model.ys}

            res, time = self.perform_iteration(sess, fetches, v_feed_dict)

            # TODO: accuracies should be weighted by batch sizes before averaging
            samples_in_batch = res['ys'].shape[0]
            total_num_samples += samples_in_batch
            valid_ys.append(res['ys'])
            valid_ts.append(v_feed_dict[self.model.ts])
            accuracies.append(res['accuracy']*samples_in_batch)

        # convert all predictions to strings and lists of words
        valid_ys = np.concatenate(valid_ys, axis=0)
        valid_ts = np.concatenate(valid_ts, axis=0)
        str_ts, str_ys = utils.numpy_to_str(valid_ts, valid_ys, self.alphabet)
        t_words, y_words = utils.strs_to_words(str_ts, str_ys)

        # compute performance metrics
        valid_acc = np.sum(accuracies)/total_num_samples
        corpus_bleu = pm.corpus_bleu(t_words, y_words)
        edit_dist = pm.mean_char_edit_distance(str_ys, str_ts)

        # print results
        self.visualize_ys(res['ys'], v_feed_dict)
        print('\t{:s}{:.2f}%'.format('accuracy:'.ljust(25), (valid_acc * 100)))
        print('\t{:s}{:.5f}'.format('BLEU:'.ljust(25), corpus_bleu))
        print('\t{:s}{:.5f}'.format('Mean edit dist per char:'.ljust(25), edit_dist))

        # Write TensorBoard summaries
        if self.summarywriter:
            feed_dict = {
                self.model.accuracy: valid_acc,
                self.bleu: corpus_bleu,
                self.edit_dist: edit_dist }
            fetches = [self.val_summaries, self.model.global_step]
            summaries, i = sess.run(fetches, feed_dict)
            self.summarywriter.add_summary(summaries, i)

        print("Continue training..")


if __name__ == '__main__':
    trainer = Trainer()
