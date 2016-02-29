## A collection of methods used in the master-code repository

def reverse_dict(d):
    return { v: k for (k, v) in d.iteritems() }


def dict_to_char(num_to_char_dict, list_to_find):
    return [ num_to_char_dict[x] for x in list_to_find ]
