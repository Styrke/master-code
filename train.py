import numpy as np

import tensorflow as tf
import model
import text_loader
from frostings.loader import *
from gen_dummy_data import get_batch

# initialize placeholders for the computation graph
X = tf.placeholder(tf.int32, shape=[None, 25], name='x_input')
t = tf.placeholder(tf.int32, shape=[None, 25], name='t_input')
X_lengths = tf.placeholder(tf.int32, shape=[None], name='x_lengths')
t_mask = tf.placeholder(tf.float32, shape=[None, 25], name='t_mask')

# build model
output_logits = model.inference(
    alphabet_size=200,
    input=X,
    input_lengths=X_lengths,
    target=t)
loss = model.loss(output_logits, t, t_mask)
train_op = model.training(loss, learning_rate=0.01)
prediction = model.prediction(output_logits)

loss_summary = tf.scalar_summary('loss', loss)

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
sample_gen = SampleGenerator(text_load_method, sample_info)
batch_info = BatchInfo(batch_size=32)
text_batch_gen = text_loader.TextBatchGenerator(sample_gen, batch_info)

# reverse dictionaries and define function for converting prediction to string
inp_alpha = {v: k for k, v in text_batch_gen.alphadict[0].iteritems()}
out_alpha = {v: k for k, v in text_batch_gen.alphadict[1].iteritems()}


def to_str(seq, alphadict):
    return ''.join([alphadict[c] for c in seq])

with tf.Session() as sess:
    # initialize parameters
    tf.initialize_all_variables().run()

    summaries = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("train", sess.graph_def)

    for i, batch in enumerate(text_batch_gen.gen_batch()):
        feed_dict = {
            X: batch['x_encoded'],
            t: batch['t_encoded'],
            X_lengths: batch['x_len'],
            t_mask: batch['t_mask']
        }

        fetches = [loss, prediction, summaries, train_op]
        res = sess.run(fetches, feed_dict=feed_dict)

        # every 10 iterations print x-sentence ::: t-prediction ::: t-truth
        if i % 10 == 0:
            for j in xrange(32):
                print '%s ::: %s ::: %s' % (
                        to_str(batch['x_encoded'][j], inp_alpha),
                        to_str(res[1][j], out_alpha),
                        to_str(batch['t_encoded'][j], out_alpha)
                    )
            writer.add_summary(res[2], i)

        # if i % 10 == 0:
        print 'Iteration %i Loss: %f' % (i, np.mean(res[0]))
