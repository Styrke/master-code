import numpy as np
from sklearn import cross_validation
try:
    import cPickled as pickle
except:
    import pickle

def create_split(n, target_path="validation_split_v1.pkl", valid_size=0.001,
    random_state=np.random.RandomState(42)):

    n_iter = 1 # hacked to only work with one split
    split = cross_validation.ShuffleSplit(n, n_iter=1,
        test_size=valid_size, random_state=random_state)
    indices_train, indices_valid = next(iter(split))

    data_to_save = {
        'indices_train': indices_train,
        'indices_valid': indices_valid,
        }
    with open(target_path, 'wb') as f:
        pickle.dump(data_to_save, f, pickle.HIGHEST_PROTOCOL)

    print('Split stored in %s' % target_path)

if __name__ == '__main__':
    create_split(100)
