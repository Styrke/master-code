import numpy as np

import tensorflow as tf
import model

with tf.Session() as sess:
    input = tf.placeholder(tf.int32)
    out = model.inference(alphabet_size=3, input=input)

    # initialize parameters
    init = tf.initialize_all_variables()
    sess.run(init)

    inp = np.array([[1,0,2],[2,0,1]])
    print sess.run(out, feed_dict={input: inp})
