import numpy as np
from collections import Counter
import time

def load_data(train='adult.data', test='adult.test'):
    file = open(train)
    file1 = open(test)
    r_X = []
    r_y = []
    r1_X = []
    r1_y = []

    for line in file.readlines():
        items = str(line).split(',')
        r_X.append(items[:-1])
        r_y.append(items[-1] == ' >50K\n')
    for line in file1.readlines():
        items = str(line).split(',')
        r1_X.append(items[:-1])
        r1_y.append(items[-1] == ' >50K.\n')
    del r_X[32561], r_y[32561]
    y = np.array(r_y).astype('int')
    X = np.array(r_X)
    y1 = np.array(r1_y).astype('int')
    X1 = np.array(r1_X)

    def count(row):
        index = dict()
        names = set()
        for i in row:
            if i not in names:
                names.add(i)
        for i in names:
            i_num = (row[y == 1] == i).sum()
            index[i] = i_num
        for i in range(len(index.keys())):
            max_index = min(index)
            row[row == max_index] = index[max_index]
            row1[row1 == max_index] = index[max_index]
            del index[max_index]
        return row.astype('float')

    for num in [1, 3, 5, 6, 7, 8, 9, 13]:
        row1 = X1[:, num]
        count(X[:, num])
        X1[:, num] = row1.astype('float')

    X = np.matrix(X.astype(float)).T
    X = X / (X.max(axis=1) - X.min(axis=1))
    X = X.T
    X1 = np.matrix(X1.astype(float)).T
    X1 = X1 / (X1.max(axis=1) - X1.min(axis=1))
    X1 = X1.T
    X_train = X
    y_train = y
    X_test = X1
    y_test = y1
    return X_train, X_test, y_train, y_test


class KNN(object):
    def __init__(self, X, y,k=3):
        self.X = X
        self.y = y
        self.k = k

    def single_predict(self, x):
        diffmat = np.repeat(x, self.X.shape[0]).reshape(
            x.shape[1], self.X.shape[0]).T - self.X
        squarediffmat = (np.matrix(np.array(diffmat)**2)).sum(axis=1)
        result = []
        for k in range(self.k):
            index = np.where(squarediffmat == np.min(squarediffmat))
            result.append(self.y[index[0][0]])
            squarediffmat[index] = float('inf')
        return Counter(result).most_common(1)[0][0]

    def predict(self, x):
        result = []
        for i in x:
            result.append(self.single_predict(i))
        return result


def main():
    print('loading data...')
    X_train, X_test, y_train, y_test = load_data()
    print('done!')
    t1=time.time()
    print('train_size:', X_train.shape[0], 'test_size:', X_test.shape[0])
    clf = KNN(X_train, y_train,k=6)
    pre_y = clf.predict(X_test)
    print('accuracy:', str((y_test == pre_y).sum() / len(y_test) * 100)[:5], '%')
    print('used_time:',str(time.time()-t1)[:5]+'s')


if __name__ == '__main__':
    main()
