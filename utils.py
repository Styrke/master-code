## A collection of methods used in the master-code repository
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import nltk
import os

nltk.download('punkt')

# for usage in printing output
def reverse_dict(d):
    return { v: k for (k, v) in d.iteritems() }


def dict_to_char(num_to_char_dict, list_to_find):
    return [ num_to_char_dict[x] for x in list_to_find ]


# --- TSNE: Look at tensorflow tutorial ---
# for usage in T-sne (to plot)
def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18)) # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    plt.savefig(filename)


# for usage in T-sne (normalized embeddings)
def get_normalized_embeddings(model):
    try:
        import tensorflow as tf
        norm = tf.sqrt(tf.reduce_sum(tf.square(
            model.embeddings), 1, keep_dims=True))
        return model.embeddings / norm
    except ImportError:
        print("Please install tensorflow")


# T-sne
def create_tsne(model, alphadict, plot_only=100, plx=20):
    """
        Takes a model (that has a variable 'embeddings'), as well as some labels.
        The embeddings needs to be normalized (for some reason).
        The embeddings are projected (some KL SGD procedure) to a lower dimension.
        The labels are then placed into the lower dimension distribution.
        All of it is plottet
    """
    normalized_embeddings = get_normalized_embeddings(model)
    final_embeddings = normalized_embeddings.eval()
    tsne = TSNE(perplexity=plx, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    reverse_dictionary = alphadict#reverse_dict(alphadict)
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)

def acc(out, label, mask):
    """ Takes an output (soft labelled/logits), labels and a mask
        outputs maksed accuracy
    """
    out = np.argmax(out, axis=2) # axis 2 is axis with predictions
    masked_sum = np.sum(((out == label).flatten()*mask.flatten())) \
                   .astype('float32')
    return  masked_sum / np.sum(mask).astype('float32')


def bleu(references, hypothesis):
    """Compute BLEU score.

    Keyword arguments:
    references -- list of strings with target translations
    hypotheses -- list of strings with hypothesis translations to test
    """
    references = [[r] for r in references]
    hypothesis = hypothesis
    return nltk.bleu_score.corpus_bleu(references, hypothesis)

def numpy_to_str(references, hypothesis, alphabet):
    """Wrapper for numpy arrays

    Keyword arguments:
    references -- numpy array of target translations
    hypotheses -- numpy array of predictions
    """
    str_ts = []
    str_ys = []
    for i in range(references.shape[0]):
        str_ts.append(alphabet.decode(references[i]) \
              .split(alphabet.eos_char)[0])
        str_ys.append(alphabet.decode(hypothesis[i]) \
              .split(alphabet.eos_char)[0])
    
    return str_ts, str_ys

def numpy_to_words(*args):
    str_ts, str_ys = numpy_to_str(*args)
    words_ts = [nltk.word_tokenize(str_t) for str_t in str_ts]
    words_ys = [nltk.word_tokenize(str_y) for str_y in str_ys]
    return words_ts, words_ys
