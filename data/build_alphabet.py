def generate_alphabet(filenames, alphabet_file):
    """Make a list of all characters that appear at least once in a file in the filenames list.
    Save the list as alphabet_file with one character per line."""
    contents = ''
    for filename in filenames:
        print("loading (%s) ..." %filename)
        with open(filename, 'r') as f:
            contents += f.read()
    print("writing ...")
    with open(alphabet_file, 'w') as f:
        f.write('\n'.join(set(contents)))
    print("done ...")

generate_alphabet(['train/europarl-v7.fr-en.en', 'train/europarl-v7.fr-en.fr'], 'alphabet')
