## A collection of methods used in the master-code repository

def reverse_dict(d):
    return { v: k for (k, v) in d.iteritems() }


def dict_to_char(alphadict, int_list_in):
    char_list_out
    for int_in in int_list_in:
        for char_dict, int_dict in alphadict.items():
            if int_dict == int_in:
                char_list_out.append(char_dict)
    return char_list_out
