import glob
import subprocess

subprocess.call('rm -rf data/train/*.norm', shell=True)
subprocess.call('rm -rf data/train/*.tok', shell=True)
subprocess.call('rm -rf data/valid/*.norm', shell=True)
subprocess.call('rm -rf data/valid/*.tok', shell=True)
subprocess.call('rm -rf data/test/*.norm', shell=True)
subprocess.call('rm -rf data/test/*.tok', shell=True)

train_paths = set(glob.glob('data/train/*'))
train_data_paths = set(glob.glob('data/train/*.tgz'))
train_paths = list(train_paths - train_data_paths)

valid_paths = set(glob.glob('data/valid/*'))
valid_data_paths = set(glob.glob('data/valid/*.tgz'))
valid_paths = list(valid_paths - valid_data_paths)

test_paths = set(glob.glob('data/test/*'))
test_data_paths = set(glob.glob('data/test/*.tgz'))
test_paths = list(test_paths - test_data_paths)

def preprocess(paths):
    paths_norm = [] # might not be nessesary to keep track in a list ...
    paths_toks =[]
    for path in paths:
        print('preprocessing: %s' %path)
        lang = path[-2:]
        command1 = 'perl data/preprocess/normalize-punctuation.perl -l'
        path_norm = path + '.norm'
        paths_norm.append(path_norm)
        call1 = '%s %s < %s > %s' % (command1, lang, path, path_norm)
        command2 = 'perl data/preprocess/tokenizer_apos.perl - threads 5 -l'
        path_toks = path + '.tok'
        paths_toks.append(path_toks)
        call2 = '%s %s < %s > %s' % (command2, lang, path_norm, path_toks)
        call3 = 'rm -rf %s' % path
        print('making: %s ...' % path_norm)
        subprocess.call(call1, shell=True)
        print('making: %s ...' % path_toks)
        subprocess.call(call2, shell=True)
        print('removing: %s ...' % path)
        subprocess.call(call3, shell=True)
    return path_toks
#print('---PREPROCESSING: train---')
#train_paths_tok = preprocess(train_paths)

print('---PREPROCESSING: valid---')
valid_paths_tok = preprocess(valid_paths)
print('---PREPROCESSING: test---')
test_paths_tok = preprocess(test_paths)

subprocess.call('rm -rf data/train/*.norm', shell=True)
subprocess.call('rm -rf data/valid/*.norm', shell=True)
subprocess.call('rm -rf data/test/*.norm', shell=True)
