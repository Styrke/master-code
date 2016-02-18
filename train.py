import numpy as np

import tensorflow as tf
import model
from gen_dummy_data import get_batch

with tf.Session() as sess:
    input = tf.placeholder(tf.int32)
    lengths = tf.placeholder(tf.int32)
    ts = tf.placeholder(tf.int64)
    out = tf.squeeze(model.inference(alphabet_size=15, input=input, lengths=lengths))
    loss = model.loss(out, ts)

    # initialize parameters
    tf.initialize_all_variables().run()

    # copute accuracy
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), ts), dtype=tf.float32))

    # optimize
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    optimizer_op = optimizer.minimize(loss)

    inp = np.array([[1, 0, 2],
                    [2, 0, 1]])
    lens = [2, 3]
    for i in xrange(100):
        targets, inputs, masks, lens = get_batch(32)
        res = sess.run([loss, acc, optimizer_op], feed_dict={input: inputs, lengths: lens, ts: targets})
        if i % 10 == 0:
            print res[1], np.mean(res[0])
