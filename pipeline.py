import numpy as np
import pickle
from time import time
from numpy.random import shuffle

symbol_map = dict([(x, chr(x)) for x in range(48, 58) + range (65, 91) + range(97, 123)])
t = time()
a = pickle.load(open('final_2000.pickle', 'rb'))
test = pickle.load(open('final_test_2000.pickle', 'rb'))
print('loaded in ', time() - t)

def next_batch(batch_size):
	shuffle(a)
	j = 0
	size = len(a)	
	while(batch_size+j < size):
		t1, t2 = zip(*a[j:j+batch_size])
		t1 = np.stack(t1, 0)
		t2 = np.stack(t2, 0)
		yield t1.astype(np.float32) / 255, t2
		j += batch_size
	t1, t2 = zip(*a[j:])
	t1 = np.stack(t1, 0)
	t2 = np.stack(t2, 0)
	yield t1.astype(np.float32) / 255, t2

def test_batch(batch_size):
	j = 0
	size = len(test)
	while(batch_size+j <= size):
		t1, t2 = zip(*test[j:j+batch_size])
		t1 = np.stack(t1, 0)
		t2 = np.stack(t2, 0)
		yield t1.astype(np.float32) / 255, t2
		j += batch_size
