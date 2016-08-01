import time
import click
import os
import numpy as np
import tensorflow as tf
import importlib
from datetime import datetime

from augmentor import Augmentor
from utils import basic as utils
import utils.performancemetrics as pm
from utils.tfhelper import run

SAVER_PATH = {'base': 'train/',
              'checkpoint': 'checkpoints/',
              'log': 'logs/',
              'test': 'test/'}


@click.command()
@click.argument('config', default='test')
class Trainer:
    """Train a translation model."""

    def __init__(self, config):
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.load_config(config)
        self.setup_reload_path()

        self.alphabet_src = self.model.get_alphabet_src()
        self.alphabet_tar = self.model.get_alphabet_tar()

        self.train()

    def setup_reload_path(self):
        self.latest_checkpoint = None
        self.checkpoint_saver = None
        self.summarywriter = None
        if not self.name:
            return  # Nothing more to do

        self.local_path        = os.path.join(SAVER_PATH['base'], self.name)
        self.summary_path = os.path.join(self.local_path, SAVER_PATH['log'])

        print("Will read and write from '{:s}' (checkpoints and logs)".format(self.local_path))

        # Prepare for saving checkpoints
        if self.save_freq:
            self.checkpoint_saver = tf.train.Saver(max_to_keep=self.model.max_to_keep)
            checkpoint_path = os.path.join(self.local_path, SAVER_PATH['checkpoint'])
            self.checkpoint_file_path = os.path.join(checkpoint_path, 'checkpoint')
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            self.latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        else:
            print("WARNING: 'save_freq' is 0 so checkpoints won't be saved!")

    def setup_validation_summaries(self):
        """A hack for recording performance metrics with TensorBoard."""
        self.bleu = tf.placeholder(tf.float32)
        self.moses_bleu = tf.placeholder(tf.float32)
        self.edit_dist = tf.placeholder(tf.float32)

        valid_summaries = [
            tf.scalar_summary('validation/loss', self.model.valid_loss),
            tf.scalar_summary('validation/accuracy', self.model.valid_accuracy),
            tf.scalar_summary('validation/bleu', self.bleu),
            tf.scalar_summary('validation/moses_bleu', self.moses_bleu),
            tf.scalar_summary('validation/edit dist per char', self.edit_dist)
        ]

        return tf.merge_summary(valid_summaries)

    def load_config(self, config_name):
        # load the config module to use
        config_path = 'configs.' + config_name
        config = importlib.import_module(config_path)

        # copy settings that affect the training script
        self.batch_size_train = config.Model.batch_size_train
        self.batch_size_valid = config.Model.batch_size_valid
        self.seq_len_x = config.Model.seq_len_x
        self.seq_len_t = config.Model.seq_len_t
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
            inp  = self.alphabet_src.decode(feed_dict[self.model.Xs][j]).ljust(self.seq_len_x)
            pred = self.alphabet_tar.decode(ys[j]).ljust(pred_len)
            targ = self.alphabet_tar.decode(feed_dict[self.model.ts][j])
            print('{1} {0} {2} {0} {3}'.format(sep, inp, pred, targ))

    def train(self):
        print("Training..")
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
            # Prepare for writing TensorBoard summaries
            if self.tb_log_freq and self.name:
                if not os.path.exists(self.summary_path) and self.tb_log_freq:
                    os.makedirs(self.summary_path)
                self.summarywriter = tf.train.SummaryWriter(self.summary_path, sess.graph)

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
                    t_feed_dict[self.model.ts_go] = augmentor.run(
                            t_feed_dict[self.model.ts_go], extra['t_len'],
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
        losses, accuracies, valid_y_strings, valid_t_strings, valid_x_strings, valid_a, valid_az, valid_ar= [], [], [], [], [], {}, {}, {}
        for v_feed_dict in self.model.next_valid_feed():
            fetches = {'accuracy': self.model.valid_accuracy,
                       'ys': self.model.valid_ys,
                       'loss': self.model.valid_loss, 
                      }
            added_dict = dict()
            if self.num_h == 1:
                added_dict['valid_a0'] = self.valid_a0
                added_dict['valid_az0'] = self.valid_az0
                added_dict['valid_ar0'] = self.valid_ar0
            if self.num_h == 5:
                added_dict['valid_a0'] = self.valid_a0
                added_dict['valid_az0'] = self.valid_az0
                added_dict['valid_ar0'] = self.valid_ar0
                added_dict['valid_a1'] = self.valid_a1
                added_dict['valid_az1'] = self.valid_az1
                added_dict['valid_ar1'] = self.valid_ar1
                added_dict['valid_a2'] = self.valid_a2
                added_dict['valid_az2'] = self.valid_az2
                added_dict['valid_ar2'] = self.valid_ar2
                added_dict['valid_a3'] = self.valid_a3
                added_dict['valid_az3'] = self.valid_az3
                added_dict['valid_ar3'] = self.valid_ar3
                added_dict['valid_a4'] = self.valid_a4
                added_dict['valid_az4'] = self.valid_az4
                added_dict['valid_ar4'] = self.valid_ar4

            res, time = self.perform_iteration(sess, fetches, v_feed_dict)

            # keep track of num_samples in batch and total
            samples_in_batch = res['ys'].shape[0]
            total_num_samples += samples_in_batch

            # convert to strings
            valid_ys, valid_ts, valid_xs = res['ys'], v_feed_dict[self.model.ts], v_feed_dict[self.model.Xs]
            str_ts, str_ys = utils.numpy_to_str(valid_ts, valid_ys, self.alphabet_tar)
            str_xs, _ = utils.numpy_to_str(valid_xs, valid_xs, self.alphabet_src)
            valid_y_strings += str_ys
            valid_t_strings += str_ts
            valid_x_strings += str_xs

            # collect loss and accuracy
            losses.append(res['loss']*samples_in_batch)
            accuracies.append(res['accuracy']*samples_in_batch)

            # initiating dictionaries for hierarchical attention tracking
            if valid_a == {}:
                if self.num_h == 1:
                    valid_a[0] = res['valid_a0'].transpose(1, 0, 2)
                    valid_az[0] = res['valid_az0'].transpose(1, 0, 2)
                    valid_ar[0] = res['valid_ar0'].transpose(1, 0, 2)
                if self.num_h == 5:
                    for i in self.num_h:
                        valid_a[i] = res['valid_a%d' % i].transpose(1, 0, 2)
                        valid_az[i] = res['valid_az%d' % i].transpose(1, 0, 2)
                        valid_ar[i] = res['valid_ar%d' % i].transpose(1, 0, 2)
            # only implemented first batch for this so far
            #else:
            #    valid_a[i].append(res['valid_a'][i].transpose(1, 0, 2))
            #    valid_az[i].append(res['valid_az'][i].transpose(1, 0, 2))
            #    valid_ar[i].append(res['valid_ar'][i].transpose(1, 0, 2))

        # convert all prediction strings to lists of words (for computing bleu)
        t_words, y_words = utils.strs_to_words(valid_y_strings, valid_t_strings)

        # compute performance metrics
        valid_loss = np.sum(losses)/total_num_samples
        valid_acc = np.sum(accuracies)/total_num_samples
        corpus_bleu = pm.corpus_bleu(t_words, y_words)
        edit_dist = pm.mean_char_edit_distance(valid_y_strings, valid_t_strings)

        # print results
        self.visualize_ys(res['ys'], v_feed_dict)
        print('\t{:s}{:.5f}'.format('loss:'.ljust(25), valid_loss))
        print('\t{:s}{:.2f}%'.format('accuracy:'.ljust(25), (valid_acc * 100)))
        print('\t{:s}{:.5f}'.format('BLEU:'.ljust(25), corpus_bleu))
        print('\t{:s}{:.5f}'.format('Mean edit dist per char:'.ljust(25), edit_dist))

        def dump_to_file(thefile, thelist):
            tot_str = '\n'.join(thelist)
            with open(thefile, "w") as file:
                file.write(tot_str)
        # TODO move this to setup
        if self.name:
            path_bleu = 'bleu/%s-%s' % (self.name, self.timestamp)
            path_a = 'attention/%s-%s' % (self.name, self.timestamp)
            path_az = 'gru-z-gate/%s-%s' % (self.name, self.timestamp)
            path_ar = 'gru-r-gate/%s-%s' % (self.name, self.timestamp)
        else:
            path_bleu = 'bleu/%s-%s' % ('no-name', self.timestamp)
            path_a = 'attention/%s-%s' % ('no-name', self.timestamp)
            path_az = 'gru-z-gate/%s-%s' % ('no-name', self.timestamp)
            path_ar = 'gru-r-gate/%s-%s' % ('no-name', self.timestamp)
        path_to_bleu =  os.path.join(SAVER_PATH['base'], path_bleu)
        path_to_a =  os.path.join(SAVER_PATH['base'], path_a)
        path_to_az =  os.path.join(SAVER_PATH['base'], path_az)
        path_to_ar =  os.path.join(SAVER_PATH['base'], path_ar)
        if not os.path.exists(path_to_bleu):
            os.makedirs(path_to_bleu)
        if not os.path.exists(path_to_a):
            os.makedirs(path_to_a)
        if not os.path.exists(path_to_az):
            os.makedirs(path_to_az)
        if not os.path.exists(path_to_ar):
            os.makedirs(path_to_ar)
        reference = os.path.join(path_to_bleu, 'reference.txt')
        translated = os.path.join(path_to_bleu, 'translated.txt')
        source = os.path.join(path_to_bleu, 'source.txt')
        a_path = os.path.join(path_to_a, 'a.npy')
        az_path = os.path.join(path_to_az, 'az.npy')
        ar_path = os.path.join(path_to_ar, 'ar.npy')
        #np.save(a_path, valid_a)
        #np.save(az_path, valid_az)
        #np.save(ar_path, valid_ar)

        if not os.path.exists(reference):
            dump_to_file(reference, valid_t_strings)
        if not os.path.exists(source):
            dump_to_file(source, valid_x_strings)
        dump_to_file(translated, valid_y_strings)
        out = pm.moses_bleu(translated, reference)
        # Write TensorBoard summaries
        if self.summarywriter:
            feed_dict = {
                self.model.valid_loss: valid_loss,
                self.model.valid_accuracy: valid_acc,
                self.bleu: corpus_bleu,
                self.moses_bleu: out,
                self.edit_dist: edit_dist }
            fetches = [self.val_summaries, self.model.global_step]
            summaries, i = sess.run(fetches, feed_dict)
            self.summarywriter.add_summary(summaries, i)

        print("Continue training..")


if __name__ == '__main__':
    trainer = Trainer()
