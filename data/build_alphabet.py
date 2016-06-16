from collections import Counter
import pickle

def generate_alphabet(filenames, alphabet_file):
    """ Creates list of tuples containing character 
    and number of occurrences in given filenames.
    It is alphabetically sorted.
    The sorted list will be pickled to disk.
    """

    contents = Counter()

    for filename in filenames:
        print("loading (%s) ..." %filename)
        with open(filename, 'r') as f:
            contents.update(f.read()
                            .replace("\r\n", "\n")
                            .replace("\r", "\n")
                            .replace("\n", ""))

    tuples_sorted_alphabetically = contents#sorted(list(contents.items()), key=lambda tup: tup[0])

    print("dumping %d tuples (char, num) to disk ..." %len(contents.items()))
    with open(alphabet_file, 'bw') as f:
        pickle.dump(tuples_sorted_alphabetically, f)
    print("done ...")

generate_alphabet(['data/train/europarl-v7.de-en.en.tok', 'data/train/commoncrawl.de-en.en.tok', 'data/train/news-commentary-v10.de-en.en.tok'], 'data/alphabet/de-en.en.tok.dict')
generate_alphabet(['data/train/europarl-v7.de-en.de.tok', 'data/train/commoncrawl.de-en.de.tok', 'data/train/news-commentary-v10.de-en.de.tok'], 'data/alphabet/de-en.de.tok.dict')
