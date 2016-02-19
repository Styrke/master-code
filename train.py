import numpy as np

import tensorflow as tf
import model
from gen_dummy_data import get_batch

with tf.Session() as sess:
    # list of sequences' length
    seq_lens = tf.placeholder(tf.int32, name='seq_lengths')

    X = tf.placeholder(tf.int32, name='input')
    t = tf.placeholder(tf.int64, name='target_truth')
    # predict
    y = tf.squeeze(
          model.inference(
              alphabet_size=15, 
              input=X, 
              lengths=seq_lens), 
          name='prediction')

    with tf.name_scope('xent'):
      loss = model.loss(y, t)
      _ = tf.scalar_summary('loss', tf.reduce_mean(loss))

    # initialize parameters
    tf.initialize_all_variables().run()

    with tf.name_scope('predictions'):
      accuracy = tf.reduce_mean(
                   tf.cast(tf.equal(tf.argmax(y, 1), t), 
                           dtype=tf.float32), 
                   name='accuracy')
      _ = tf.scalar_summary('accuracy', accuracy)

    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # merge summaries and add writer
    summaries = tf.merge_all_summaries()
    writer    = tf.train.SummaryWriter('/tmp/test_logs', sess.graph_def)

    for i in xrange(100):
        train_t, train_X, _, lens = get_batch(32)
        feed_dict = {X: train_X, seq_lens: lens, t: train_t}
        res = sess.run([loss, accuracy, optimizer, summaries], 
                       feed_dict=feed_dict)
        # write summaries
        writer.add_summary(res[3], i) 
        if i % 10 == 0:
            print res[1], np.mean(res[0])
