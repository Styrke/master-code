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

generate_alphabet(['data/train/europarl-v7.fr-en.en'], 'data/alphabet/dict_europarl.fr-en.en')
generate_alphabet(['data/train/europarl-v7.fr-en.fr'], 'data/alphabet/dict_europarl.fr-en.fr')

generate_alphabet(['data/train/europarl-v7.da-en.en'], 'data/alphabet/dict_europarl.da-en.en')
generate_alphabet(['data/train/europarl-v7.da-en.da'], 'data/alphabet/dict_europarl.da-en.da')

generate_alphabet(['data/train/europarl-v7.de-en.en'], 'data/alphabet/dict_europarl.de-en.en')
generate_alphabet(['data/train/europarl-v7.de-en.de'], 'data/alphabet/dict_europarl.de-en.de')

generate_alphabet(['data/train/europarl-v7.de-en.en', 'data/train/commoncrawl.de-en.en', 'data/train/news-commentary-v10.de-en.en'], 'data/alphabet/dict_wmt.de-en.en')
generate_alphabet(['data/train/europarl-v7.de-en.de', 'data/train/commoncrawl.de-en.de', 'data/train/news-commentary-v10.de-en.de'], 'data/alphabet/dict_wmt.de-en.de')

generate_alphabet(['data/train/europarl-v7.de-en.en.norm', 'data/train/commoncrawl.de-en.en.norm', 'data/train/news-commentary-v10.de-en.en.norm'], 'data/alphabet/dict_wmt_norm.de-en.en')
generate_alphabet(['data/train/europarl-v7.de-en.de.norm', 'data/train/commoncrawl.de-en.de.norm', 'data/train/news-commentary-v10.de-en.de.norm'], 'data/alphabet/dict_wmt_norm.de-en.de')

generate_alphabet(['data/train/europarl-v7.de-en.en.tok', 'data/train/commoncrawl.de-en.en.tok', 'data/train/news-commentary-v10.de-en.en.tok'], 'data/alphabet/dict_wmt_tok.de-en.en')
generate_alphabet(['data/train/europarl-v7.de-en.de.tok', 'data/train/commoncrawl.de-en.de.tok', 'data/train/news-commentary-v10.de-en.de.tok'], 'data/alphabet/dict_wmt_tok.de-en.de')
