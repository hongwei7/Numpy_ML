print('loading data...')
import numpy as np
file = open('adult.data')
r_X = []
r_y = []
for line in file.readlines():
    items = str(line).split(',')
    r_X.append(items[:-1])
    r_y.append(items[-1] == ' >50K\n')
del r_X[32561], r_y[32561]
y = np.array(r_y).astype('int')
X = np.array(r_X)


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
        del index[max_index]
    return row.astype('float')


for num in [1, 3, 5, 6, 7, 8, 9, 13]:
    count(X[:, num])
X = np.matrix(X.astype(float)).T
X = X / (X.max(axis=1) - X.min(axis=1))
print('done!')
X, y = X, y


class node(object):
    def __init__(self, dataset, depth=0, parent=None, median=None):
        self.parent = parent
        self.median = median
        self.depth = depth
        self.data = dataset
        self.create_child_node()

    def create_child_node(self):
        if self.data.shape[1] <= 1:
            if self.data.shape[1] == 1:
                self.value = self.data
            else:
                self.value = None
            return
        else:
            self.median = np.median(self.data[(self.depth)], axis=1)
            median = self.median
            left = self.data[np.repeat(self.data[self.depth, :] < median, self.data.shape[0]).reshape(
                self.data.shape[1], self.data.shape[0]).T]
            left = left.reshape(self.data.shape[0], int(
                left.shape[1] / self.data.shape[0]))
            right = self.data[np.repeat(self.data[self.depth, :] >= median, self.data.shape[0]).reshape(
                self.data.shape[1], self.data.shape[0]).T]
            right = right.reshape(self.data.shape[0], int(
                right.shape[1] / self.data.shape[0]))
            self.left = node(depth=(self.depth + 1) %
                             self.data.shape[0], parent=self, dataset=left)
            self.right = node(depth=(self.depth + 1) %
                              self.data.shape[0], parent=self, dataset=right)
            self.value = self.data[:, 0]


class kdd_KNN(object):
    def fit(self, X, y):
        self.X=X
        self.kdtree = node(self.X)

    def predict(self, x):
        x = x.T
        result = []
        i = 0
        for xi in x:
            i += 1
            xi = xi.T
            cloest = self.find(xi, self.kdtree)
            result.append(
                y[np.where((self.X == cloest).sum(axis=0) == xi.shape[0])[1]][0])
        return result

    def find(self, x, kdtree,depth=0):
        root = kdtree
        locate = kdtree
        cloest = locate.value
        side = None
        while locate.data.shape[1] > 1:
            if x[depth % x.shape[0]] < locate.median:
                locate = locate.left
                side = 'l'
            else:
                locate = locate.right
                side = 'r'
            depth+=1
            while locate.data.shape[1] == 0:
                locate = locate.parent
                if side == 'r':
                    locate = locate.left
                else:
                    locate = locate.right
            if (np.array(locate.value - x)**2).sum() < (np.array(cloest - x)**2).sum() and locate.value.shape[1] != 0:
                cloest = locate.value
        cloest = locate.data
        while(locate != root):
            locate = locate.parent
            depth-=1
            if side == 'r':
                other = self.find(x, locate.left,depth+1)
            else:
                other = self.find(x, locate.right,depth+1)

            if (np.array(other - x)**2).sum() < (np.array(cloest - x)**2).sum() and other.shape[1] != 0:
                cloest = other
        return cloest

if __name__ == '__main__':
    import random
    import time
    t1=time.time()
    k=random.randint(1000,8500)
    d=k+100
    m=250
    from sklearn.neighbors import KNeighborsClassifier
    clf=KNeighborsClassifier(n_neighbors=1)
    clf.fit(X[:, :m].T, y[:m])
    pre_y=clf.predict(X[:, k:d].T)
    print(list(pre_y),'\n',(pre_y==y[k:d]).sum()/(d-k)*100,'%')
    print(list(y[k:d]))
    t3=time.time()
    print(time.time()-t1,'s')
    clf = kdd_KNN()
    clf.fit(X[:, :m], y)
    #print(X[:,k:d])
    pre_y=clf.predict(X[:, k:d])
    print(list(pre_y),'\n',(pre_y==y[k:d]).sum()/(d-k)*100,'%')
    print(list(y[k:d]))
    t2=time.time()
    print(t2-t3,'s')






