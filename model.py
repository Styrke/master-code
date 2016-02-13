import tensorflow as tf
import numpy as np
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops import control_flow_ops

BATCH_SIZE = 2
EMBEDDING_SIZE = 3
ENCODER_HIDDEN_SIZE = 10
ENCODER_OUTPUT_SIZE = 10


def inference(alphabet_size, input):
    # character embeddings
    embeddings = tf.random_uniform(shape=[alphabet_size, EMBEDDING_SIZE])
    embedded = tf.gather(embeddings, input)

    shape = tf.shape(embedded)
    length = shape[1]  # number of loop iterations

    W_ih = tf.truncated_normal([EMBEDDING_SIZE, ENCODER_HIDDEN_SIZE])
    W_hh = tf.truncated_normal([ENCODER_HIDDEN_SIZE, ENCODER_HIDDEN_SIZE])
    W_ho = tf.truncated_normal([ENCODER_HIDDEN_SIZE, ENCODER_OUTPUT_SIZE])

    # Define loop condition and loop body
    def cond(i, inputs, hidden, output):
        return tf.less(i, length)

    def body(i, inputs, hidden, output):
        input = inputs.read(i)
        hidden = tf.tanh(tf.matmul(hidden, W_hh) + tf.matmul(input, W_ih))
        output = tf.matmul(hidden, W_ho)
        return [tf.add(i, 1), inputs, hidden, output]

    # Initialize loop variables
    i = tf.constant(1)
    # transpose so the TensorArray divides the tensor by the time dimension
    transp = tf.transpose(embedded, perm=[1, 0, 2])
    inputs = TensorArray(tf.float32, length).unpack(transp)
    hidden = tf.zeros([BATCH_SIZE, ENCODER_HIDDEN_SIZE])
    output = tf.constant(1.)

    loop_vars = [i, inputs, hidden, output]
    [i, inputs, hidden, output] = control_flow_ops.While(cond, body, loop_vars)

    return output
