def run(session, fetches, feed_dict):
    """Wrapper for making Session.run() more user friendly.

    With this function, fetches can be either a list or a dictionary.

    If fetches is a list, this function will behave like
    tf.session.run() and return a list in the same order as well. If
    fetches is a dict then this function will also return a dict where
    the returned values are associated with the corresponding keys from
    the fetches dict.

    Keyword arguments:
    session -- An open TensorFlow session.
    fetches -- A list or dict of ops to fetch.
    feed_dict -- The dict of values to feed to the computation graph.
    """
    if isinstance(fetches, list):
        return session.run(fetches, feed_dict)
    elif isinstance(fetches, dict):
        keys, values = fetches.keys(), list(fetches.values())
        res = session.run(values, feed_dict)
        return {key: value for key, value in zip(keys, res)}
    raise TypeError('Fetches of type %s not supported.' % type(fetches))
