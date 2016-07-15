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

class Predict:
    """Makes a translation given a config and a checkpoint"""

    def __init__(self, config):
        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.load_config(config)
        self.load_path()

        self.alphabet_src = self.model.get_alphabet_src()
        self.alphabet_tar = self.model.get_alphabet_tar()

        self.predict()

    def load_config(self, config_name):
        # load config module to use
        config_path = 'configs.' + config_name
        config = importlib.import_module(config_path)

        # copy settings that affects the prediction script
        self.batch_size = config.Model.batch_size_valid
        self.seq_len_x = config.Model.seq_len_x
        self.seq_len_t = config.Model.seq_len_t
        self.name = config.Model.name

        # self.config as an alias for the model
        self.config = self.model = config.Model()

    def load_path(self):
        if not self.name:
            sys.exit('No Snows here, must have a family (config) name!')
        self.local_path = os.path.join(SAVER_PATH['base'], self.name)
        print("Will read from '{:s}' (logs)".format(self.local_path))

        self.checkpoint_saver = tf.train.Saver()
        checkpoint_path = os.path.join(self.local_path, SAVER_PATH['checkpoint'])
        self.checkpoint_file_path = os.path.join(checkpoint_path, 'checkpoint')
        self.latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
        if self.latest_checkpoint is None:
            sys.exit('Girl is no-one, latest checkpoint is no-one')

    def predict(self):
        gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
            self.checkpoint_saver.restore(sess, self.latest_checkpoint)

            combined_time = 0.0
            self.validate(sess)

    def visualize_ys(self, ys, feed_dict):
        sep = ":::"
        pred_len = len(max(ys, key=len)) # length of longest predicted string
        for j in range(feed_dict[self.model.Xs].shape[0]):
            inp  = self.alphabet_src.decode(feed_dict[self.model.Xs][j]).ljust(self.seq_len_x)
            pred = self.alphabet_tar.decode(ys[j]).ljust(pred_len)
            targ = self.alphabet_tar.decode(feed_dict[self.model.ts][j])
            print('{1} {0} {2} {0} {3}'.format(sep, inp, pred, targ))

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
        losses, accuracies, valid_y_strings, valid_t_strings, valid_x_strings, attention_tracker = [], [], [], [], [], []
        for p_feed_dict in self.model.next_test_feed():
            fetches = {'accuracy': self.model.valid_accuracy,
                       'ys': self.model.valid_ys,
                       'loss': self.model.valid_loss,
                       'attention_tracker': self.model.valid_attention_tracker}

            res, time = self.perform_iteration(sess, fetches, p_feed_dict)

            # keep track of num_samples in batch and total
            samples_in_batch = res['ys'].shape[0]
            total_num_samples += samples_in_batch

            # convert to strings
            valid_ys, valid_ts, valid_xs = res['ys'], p_feed_dict[self.model.ts], p_feed_dict[self.model.Xs]
            str_ts, str_ys = utils.numpy_to_str(valid_ts, valid_ys, self.alphabet_tar)
            str_xs, _ = utils.numpy_to_str(valid_xs, valid_xs, self.alphabet_src)
            valid_y_strings += str_ys
            valid_t_strings += str_ts
            valid_x_strings += str_xs

            # collect loss and accuracy
            losses.append(res['loss']*samples_in_batch)
            accuracies.append(res['accuracy']*samples_in_batch)
            attention_tracker.append(res['attention_tracker'].transpose(1, 0, 2))

        # convert all prediction strings to lists of words (for computing bleu)
        t_words, y_words = utils.strs_to_words(valid_y_strings, valid_t_strings)

        # compute performance metrics
        valid_loss = np.sum(losses)/total_num_samples
        valid_acc = np.sum(accuracies)/total_num_samples
        corpus_bleu = pm.corpus_bleu(t_words, y_words)
        edit_dist = pm.mean_char_edit_distance(valid_y_strings, valid_t_strings)

        # print results
        self.visualize_ys(res['ys'], p_feed_dict)
        print('\t{:s}{:.5f}'.format('loss:'.ljust(25), valid_loss))
        print('\t{:s}{:.2f}%'.format('accuracy:'.ljust(25), (valid_acc * 100)))
        print('\t{:s}{:.5f}'.format('BLEU:'.ljust(25), corpus_bleu))
        print('\t{:s}{:.5f}'.format('Mean edit dist per char:'.ljust(25), edit_dist))

        def dump_to_file(thefile, thelist):
            tot_str = '\n'.join(thelist)
            with open(thefile, "w") as file:
                file.write(tot_str)
        path_bleu = 'bleu/%s-%s' % (self.name, self.timestamp)
        path_attention = 'attention/%s-%s' % (self.name, self.timestamp)
        path_to_bleu =  os.path.join(SAVER_PATH['test'], path_bleu)
        path_to_attention =  os.path.join(SAVER_PATH['test'], path_attention)
        if not os.path.exists(path_to_bleu):
            os.makedirs(path_to_bleu)
        if not os.path.exists(path_to_attention):
            os.makedirs(path_to_attention)
        reference = os.path.join(path_to_bleu, 'reference.txt')
        translated = os.path.join(path_to_bleu, 'translated.txt')
        source = os.path.join(path_to_bleu, 'source.txt')
        attention_path = os.path.join(path_to_attention, 'attention.npy')
        np.save(attention_path, attention_tracker[0])

        # dumping translations and references
        dump_to_file(reference, valid_t_strings)
        dump_to_file(source, valid_x_strings)
        dump_to_file(translated, valid_y_strings)
        out = pm.moses_bleu(translated, reference)

if __name__ == '__main__':
    predict = Predict()
