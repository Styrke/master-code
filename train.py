import numpy as np

import tensorflow as tf
import model
from gen_dummy_data import get_batch

with tf.Session() as sess:
    # list of sequences' length
    seq_lens = tf.placeholder(tf.int32, name='seq_lengths')

    X = tf.placeholder(tf.int32, shape=[32, 5], name='input')
    t = tf.placeholder(tf.int32, shape=[32, 5], name='target_truth')
    # predict
    _, loss = model.inference(
                  alphabet_size=15,
                  input=X,
                  target=t)

    # initialize parameters
    tf.initialize_all_variables().run()

    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    for i in xrange(100):
        train_t, train_X, _, lens = get_batch(32)
        feed_dict = {X: train_X, t: train_X}
        res = sess.run([loss, optimizer], 
                       feed_dict=feed_dict)
        print res[1], np.mean(res[0])
