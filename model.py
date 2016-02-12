import tensorflow as tf
from tensorflow.python.ops.tensor_array_ops import TensorArray

EMBEDDING_SIZE = 3

def inference(alphabet_size, input):
    # character embeddings
    embeddings = tf.random_uniform(shape=[alphabet_size, EMBEDDING_SIZE])
    embedded = tf.gather(embeddings, input)

    shp = tf.shape(embedded)

    return embedded
