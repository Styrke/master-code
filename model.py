import tensorflow as tf
import numpy as np
from tensorflow.python.ops.tensor_array_ops import TensorArray
from tensorflow.python.ops import control_flow_ops

EMBEDDING_SIZE = 3

def inference(alphabet_size, input):
    # character embeddings
    embeddings = tf.random_uniform(shape=[alphabet_size, EMBEDDING_SIZE])
    embedded = tf.gather(embeddings, input)

    size = tf.shape(embedded)[1] # number of loop iterations

    # Define loop condition and loop body
    def cond(i, inputs, output):
        return tf.less(i, size)

    def body(i, inputs, output):
        output = inputs.read(i)
        return [tf.add(i, 1), inputs, output]

    # Initialize loop variables
    i = tf.constant(1)
    # transpose so the TensorArray divides the tensor by the time dimension
    transp = tf.transpose(embedded, perm=[1, 0, 2])
    inputs = TensorArray(tf.float32, size).unpack(transp)
    output = tf.constant(1.)

    [i, inputs, output] = control_flow_ops.While(cond, body, [i, inputs, output])

    return output
