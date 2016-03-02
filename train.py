import numpy as np

import tensorflow as tf
from model import Model
import text_loader
from frostings.loader import *
from gen_dummy_data import get_batch

# initialize placeholders for the computation graph
Xs = tf.placeholder(tf.int32, shape=[None, 25], name='X_input')
ts = tf.placeholder(tf.int32, shape=[None, 25], name='t_input')
X_len = tf.placeholder(tf.int32, shape=[None], name='X_len')
t_mask = tf.placeholder(tf.float32, shape=[None, 25], name='t_mask')

# build model
model = Model(alphabet_size=170)
model.build(Xs, X_len, ts)
model.build_loss(ts, t_mask)
model.build_prediction()
model.training(learning_rate=0.01)

loss_summary = tf.scalar_summary('loss', model.loss)

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
text_load_method = text_loader.TextLoadMethod()
sample_info = SampleInfo(len(text_load_method.samples))
sample_gen = SampleGenerator(text_load_method, sample_info, repeat=True)
batch_info = BatchInfo(batch_size=32)
text_batch_gen = text_loader.TextBatchGenerator(sample_gen, batch_info)

# reverse dictionaries and define function for converting prediction to string
alphabet = {v: k for k, v in text_batch_gen.alphabet.iteritems()}

saver = tf.train.Saver()


def to_str(seq, alphadict):
    return ''.join([alphadict[c] for c in seq])

with tf.Session() as sess:
    # restore or initialize parameters
    latest_checkpoint = tf.train.latest_checkpoint('train/checkpoints')
    if latest_checkpoint:
        saver.restore(sess, latest_checkpoint)
    else:
        tf.initialize_all_variables().run()

    summaries = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("train/logs", sess.graph_def)

    for i, batch in enumerate(text_batch_gen.gen_batch()):
        feed_dict = {
            Xs: batch['x_encoded'],
            ts: batch['t_encoded'],
            X_len: batch['x_len'],
            t_mask: batch['t_mask']
        }

        fetches = [model.loss, model.ys, summaries, model.train_op]
        res = sess.run(fetches, feed_dict=feed_dict)

        # every 10 iterations print x-sentence ::: t-prediction ::: t-truth
        if i % 10 == 0:
            for j in xrange(32):
                print '%s ::: %s ::: %s' % (
                        to_str(batch['x_encoded'][j], alphabet),
                        to_str(res[1][j], alphabet),
                        to_str(batch['t_encoded'][j], alphabet)
                    )
            writer.add_summary(res[2], i)
            saver.save(sess,
                       'train/checkpoints/checkpoint',
                       global_step=model.global_step)

        # if i % 10 == 0:
        print 'Iteration %i Loss: %f' % (i, np.mean(res[0]))
