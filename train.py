import click
import numpy as np
import tensorflow as tf
from frostings.loader import *

import text_loader
from model import Model
from utils import acc, create_tsne as TSNE
from dummy_loader import *

use_logged_weights = False


@click.command()
@click.option(
    '--loader', type=click.Choice(['europarl', 'normal', 'talord',
    'talord_caps1', 'talord_caps2', 'talord_caps3']), default='normal',
    help='Choose dataset to load. (default: normal)')
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
@click.option('--valid-freq', default=20,
    help='Validate every N iterations. 0 to disable. (default: 20)')
@click.option('--seq-len', default=50,
    help='Maximum length of char sequences')
def train(loader, tsne, visualize, log_freq, save_freq, iterations,
          valid_freq, seq_len):
    """Train a translation model."""
    # initialize placeholders for the computation graph
    Xs = tf.placeholder(tf.int32, shape=[None, seq_len], name='X_input')
    ts = tf.placeholder(tf.int32, shape=[None, seq_len], name='t_input')
    ts_go = tf.placeholder(tf.int32, shape=[None, seq_len], name='t_input_go')
    X_len = tf.placeholder(tf.int32, shape=[None], name='X_len')
    t_mask = tf.placeholder(tf.float32, shape=[None, seq_len], name='t_mask')

    # build model
    model = Model(
        alphabet_size=337,
        max_x_seq_len=seq_len,
        max_t_seq_len=seq_len)
    model.build(Xs, X_len, ts_go)
    model.build_loss(ts, t_mask)
    model.build_prediction()
    model.training(learning_rate=0.1)

    loss_summary = tf.scalar_summary('loss', model.loss)

    # Add TensorBoard summaries to biases and weights from encoder and decoder
    for var in tf.all_variables():
        if var.name == 'rnn_encoder/BasicRNNCell/Linear/Matrix:0':
            tf.histogram_summary('weights/encoder', var)
        if var.name == 'rnn_decoder/BasicRNNCell/Linear/Matrix:0':
            tf.histogram_summary('weights/decoder', var)
        if var.name == 'rnn_encoder/BasicRNNCell/Linear/Bias:0':
            tf.histogram_summary('bias/encoder', var)
        if var.name == 'rnn_decoder/BasicRNNCell/Linear/Bias:0':
            tf.histogram_summary('bias/decoder', var)

    # initialize data loader
    # the magic number arguments in dummy loaders are for max len and
    # max spaces.

    if loader == 'europarl':
        print('Using europarl loader')
        train_load_method = text_loader.TextLoadMethod(
            ['data/train/europarl-v7.fr-en.en'],
            ['data/train/europarl-v7.fr-en.fr'], seq_len=seq_len)
        train_sample_gen = SampleGenerator(train_load_method, repeat=True)
        # data loader for eval, notices repeat = false
        train_eval_sample_gen = SampleGenerator(train_load_method, repeat=False)
    elif loader == 'normal':
        print('Using normal dummy loader')
        sample_gen = DummySampleGenerator(dummy_sampler, seq_len/6, 1, 'normal')
    elif loader == 'talord':
        print('Using talord dummy loader')
        sample_gen = DummySampleGenerator(dummy_sampler, seq_len/6, 1, 'talord')
    elif loader == 'talord_caps1':
        print('Using talord_caps1 dummy loader')
        sample_gen = DummySampleGenerator(dummy_sampler, seq_len/6, 1, 'talord_caps1')
    elif loader == 'talord_caps2':
        print('Using talord_caps2 dummy loader')
        sample_gen = DummySampleGenerator(dummy_sampler, seq_len/6, 1, 'talord_caps2')
    elif loader == 'talord_caps3':
        print('Using talord_caps3 dummy loader')
        sample_gen = DummySampleGenerator(dummy_sampler, seq_len/6, 1, 'talord_caps3')
    else:
        print('This should not happen, contact administrator')
        assert False

    train_batch_gen = text_loader.TextBatchGenerator(
        train_sample_gen, batch_size=32, seq_len=seq_len)
    # again, for evaluation purposes
    train_eval_batch_gen = text_loader.TextBatchGenerator(
        train_eval_sample_gen, batch_size=32, seq_len=seq_len)
    alphabet = train_batch_gen.alphabet

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # restore or initialize parameters
        if use_logged_weights:
            latest_checkpoint = tf.train.latest_checkpoint('train/checkpoints')
        else:
            latest_checkpoint = False  # could be more pretty
        if latest_checkpoint:
            saver.restore(sess, latest_checkpoint)
        else:
            tf.initialize_all_variables().run()

        if tsne:
            TSNE(model, alphabet.decode_dict)

        summaries = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("train/logs", sess.graph_def)

        for i, batch in enumerate(train_batch_gen.gen_batch()):
            if valid_freq and i % valid_freq == 0:
                print()
                print('Validating')
                # subset for printing purposes
                subsets = ['train']
                # giving the generator to the subset
                gens = [train_eval_batch_gen]

                for subset, gen in zip(subsets, gens):
                    print('  %s set' % subset)
                    # holders for subset (to compute acc() later!)
                    outputs = []
                    labels = []
                    masks = []
                    for valid_batch in gen.gen_batch():
                        # running the model only for inference
                        feed_dict = {
                            Xs: valid_batch['x_encoded'],
                            ts_go: valid_batch['t_encoded_go'],
                            X_len: valid_batch['x_len']
                        }

                        fetches = [model.out_tensor]
                        res = sess.run(fetches, feed_dict=feed_dict)
                        out = res[0]
                        # appending for each batch
                        outputs.append(out)
                        labels.append(valid_batch['t_encoded'])
                        masks.append(valid_batch['t_mask'])
                    # stacking all batches, list -> nd.array
                    outputs = np.vstack(outputs)
                    labels = np.vstack(labels)
                    masks = np.vstack(masks)
                    # getting validation
                    valid_acc = acc(outputs, labels, masks)
                    print('    acc:\t%.2f%%' % (valid_acc * 100))
                    print()

            feed_dict = {
                Xs: batch['x_encoded'],
                ts: batch['t_encoded'],
                ts_go: batch['t_encoded_go'],
                X_len: batch['x_len'],
                t_mask: batch['t_mask']
            }

            fetches = [model.loss, model.ys, summaries, model.train_op,
                model.out_tensor]
            res = sess.run(fetches, feed_dict=feed_dict)

            # Maybe visualize predictions
            if visualize:
                if i % visualize == 0:
                    for j in range(32):
                        click.echo('%s ::: %s ::: %s' % (
                                alphabet.decode(batch['x_encoded'][j]),
                                alphabet.decode(res[1][j]),
                                alphabet.decode(batch['t_encoded'][j])
                            ))

            # Write summaries for TensorBoard.
            writer.add_summary(res[2], i)

            if save_freq and i:
                if i % save_freq == 0:
                    saver.save(sess,
                               'train/checkpoints/checkpoint',
                               global_step=model.global_step)

            if log_freq:
                if i % log_freq == 0:
                    batch_acc = acc(res[4], batch['t_encoded'],
                        batch['t_mask'])
                    click.echo('Iteration %i Loss: %f Acc: %f' % (
                        i, np.mean(res[0]), batch_acc))

            if i >= iterations:
                click.echo('reached max iteration: %d' % i)
                break

if __name__ == '__main__':
    train()
