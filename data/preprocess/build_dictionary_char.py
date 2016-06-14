import pickle as pkl
import fileinput
import numpy
import sys
import codecs

from collections import OrderedDict


short_list = 300

def main():
    word_freqs = OrderedDict()
    lang = sys.argv[1]
    for filename in sys.argv[2:]:
        print('Processing %s' % filename)
        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip()
                #words_in = list(words_in.decode('utf8'))
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1

    words = list(word_freqs.copy().keys())
    freqs = list(word_freqs.copy().values())

    sorted_idx = numpy.argsort(freqs)
    sorted_words = [words[ii] for ii in sorted_idx[::-1]]

    worddict = OrderedDict()
    worddict['eos'] = 0
    worddict['UNK'] = 1

    if short_list is not None:
        for ii in range(min(short_list, len(sorted_words))):
            worddict[sorted_words[ii]] = ii + 2
    else:
        for ii, ww in enumerate(sorted_words):
            worddict[ww] = ii + 2

    print(worddict)
    char_keys = list(worddict.copy().keys())
    dump = ''
    for char in char_keys:
        dump += char+'\n'
    print(dump)
    print(dump.strip())
    print(len(dump))
    print(len(dump.strip()))
    print(len(sorted_words))
    assert False
    with open('data/%s.%d.dict' % (lang, short_list), 'wb') as f:
        for char in char_keys:
            print(char, file=f)

    print('Done')
    print(len(worddict))

if __name__ == '__main__':
    main()
