import tensorflow as tf
import numpy as np
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops import control_flow_ops

BATCH_SIZE = 2
EMBEDDING_SIZE = 3
MAX_SEQ_LENGTH = 5


def inference(alphabet_size, input, lengths):
    # character embeddings
    # embeddings = tf.random_uniform(shape=[alphabet_size, EMBEDDING_SIZE])
    # embedded = tf.gather(embeddings, input)
    embedded = input

    pad_size = tf.reshape(MAX_SEQ_LENGTH - tf.shape(embedded)[1], [1, 1])
    paddings = tf.pad(pad_size, np.array([[1, 0], [1, 0]]))
    rnn_pad = tf.pad(embedded,
                     paddings)

    inputs = tf.split(1, MAX_SEQ_LENGTH, rnn_pad)
    outputs = list()

    for i in xrange(MAX_SEQ_LENGTH):
        outputs.append(tf.select(tf.less(i, lengths),
                                 inputs[i],
                                 tf.zeros([2, 1], dtype=tf.int32)))

    output = tf.pack(outputs)
    # output = paddings

    return output
