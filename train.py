import numpy as np

import tensorflow as tf
import model
from gen_dummy_data import get_batch

with tf.Session() as sess:
    input = tf.placeholder(tf.int32)
    lengths = tf.placeholder(tf.int32)
    ts = tf.placeholder(tf.int64)
    out = tf.squeeze(model.inference(alphabet_size=15, input=input, lengths=lengths))

    with tf.name_scope('xent'):
      loss = model.loss(out, ts)
      _ = tf.scalar_summary('loss', tf.reduce_mean(loss))
    # initialize parameters
    tf.initialize_all_variables().run()

    with tf.name_scope('accuracy'):
      # compute accuracy
      acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), ts), dtype=tf.float32))
      _ = tf.scalar_summary('accuracy', acc)


    # optimize
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    optimizer_op = optimizer.minimize(loss)

    # merge summaries and add writer
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('/tmp/test_logs', sess.graph_def)

    for i in xrange(100):
        targets, inputs, masks, lens = get_batch(32)
        res = sess.run([loss, acc, optimizer_op, merged], feed_dict={input: inputs, lengths: lens, ts: targets})
        if i % 10 == 0:
            print res[1], np.mean(res[0])
        writer.add_summary(res[3], i)
