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
del data, label
print 'loading complete'
unique_label = label_array
unique_label = unique_label.reshape(-1, 1)
# print unique_label
ohe = OneHotEncoder()
ohe.fit(unique_label)
encoded_to_map = ohe.active_features_
dataset = []
size = 2000


def find(char):
    # print np.where(label_array == char)
    return np.where(label_array == char)

def train_data():
    for i in set(label):
        indexs = find(i)[0]
        # print len(indexs)
        if len(indexs) >= size:
            # print 'check'
            indexs = indexs[:size]
            # len(indexs)
            for j, k in zip(data_array[indexs], label_array[indexs]):
                temp = ohe.transform([[k]]).toarray()[0]
                dataset.append([j.flatten(), temp])
        else:
            for j, k in zip(data_array[indexs], label_array[indexs]):
                temp = ohe.transform([[k]]).toarray()[0]
                dataset.append([j.flatten(), temp])
    	print 'Class ', i, 'completed'

    print 'For loop finished'
    # data, label = zip(*dataset[:2])
    # print data
    # print label
    # print data[0]
    # print label[0]
    # print type(data[0])
    # print type(label[0])
    # for i in dataset[::2000]:
        # print i[1],

    # dataset = np.array(dataset)
    with open('final_2000.pickle', 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

    with open('encoded_to_map.pickle', 'wb') as g:
        pickle.dump(encoded_to_map, g, pickle.HIGHEST_PROTOCOL)

    print 'changed'
    temp = np.stack(data, axis = 0)
    print temp
    print type(temp)
    temp = np.stack(label)
    print temp
    print type(temp)
    print np.stack(label, axis = 0)

def test_data():
    for i in set(label):
        indexs = find(i)[0]
        indexs = indexs[size:size+100]
        for j, k in zip(data_array[indexs], label_array[indexs]):
            temp = ohe.transform([[k]]).toarray()[0]
            dataset.append([j.flatten(), temp])
    	print 'Class ', i, 'completed'

    with open('final_test_2000.pickle', 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
test_data()
