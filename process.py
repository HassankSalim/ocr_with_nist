import numpy as np
import pickle
# from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from time import time

t = time()
arr = pickle.load(open('SD19_28x28_402953.pickle', 'rb'))
print time() - t
data, label = zip(*arr)
label_array = np.array(label)
data_array = np.array(data)
print 'loading complete'
unique_label = label_array
unique_label = unique_label.reshape(-1, 1)
# print unique_label
ohe = OneHotEncoder()
ohe.fit(unique_label)
encoded_to_map = ohe.active_features_
dataset = []



def find(char):
    # print np.where(label_array == char)
    return np.where(label_array == char)

for i in set(label):
    indexs = find(i)[0]
    print len(indexs)
    if len(indexs) >= 5000:
        print 'check'
        indexs = indexs[:5000]
        len(indexs)
        for j, k in zip(data_array[indexs], label_array[indexs]):
            temp = ohe.transform([[k]]).toarray()[0]
            # print temp, k
            dataset.append([j, temp])
    else:
        for j, k in zip(data_array[indexs], label_array[indexs]):
            temp = ohe.transform([[k]]).toarray()[0]
            dataset.append([j, temp])

data, label = zip(*dataset)
# for i in dataset[::2000]:
    # print i[1],

with open('final.pickle', 'wb') as f:
    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

with open('encoded_to_map.pickle', 'wb') as g:
    pickle.dump(encoded_to_map, g, pickle.HIGHEST_PROTOCOL)
