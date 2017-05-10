import numpy as np
import pickle
from time import time
from numpy.random import shuffle

symbol_map = dict([(x, chr(x)) for x in range(48, 58) + range (65, 91) + range(97, 123)])
t = time()
a = pickle.load(open('final.pickle', 'rb'))
print(time() - t)
print('loaded')

def next_batch(batch_size):
	numpy.random.shuffle(a)
	j = 0
    size = len(a)
	while(batch_size+j < size):
		yield a[j:j+batch_size]
		j += batch_size
	yield a[j:]
