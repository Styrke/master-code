import gzip
import os
import numpy as np

def load_gz(path):
	if path.endswith(".gz"):
		f = gzip.open(path, 'rb')
		return np.load(f)
	else:
		return np.load(path)

def save_gz(path, data):
	np.save(path, data)
	os.system('gzip ' + path)
