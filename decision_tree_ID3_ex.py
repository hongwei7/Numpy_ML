from collections import Counter
import numpy as np


class s_data(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class node(object):
    def __init__(self, data, index_list):
        self.data = data
        self.clas = Counter(data.y).most_common()[0][0]
        self.index_list = index_list


class condition_node(object):
    def __init__(self, f_index, index_list, data):
        self.data = data
        self.f_index = f_index
        self.index_list = index_list
        self.clas = Counter(data.y).most_common()[0][0]

    def create_nodes(self, data):
        self.branches = dict()
        for f in set(data.x.T[self.f_index]):
            self.branches[f] = create_tree(s_data(
                data.x[data.x.T[self.f_index] == f], data.y[data.x.T[self.f_index] == f]), self.index_list)

    def judge(self, x):
        return self.data_list[x[self.f_index]]


def claculate_H_D(data):
    H_D = 0
    for yi in set(data.y):
        Ck_D = (data.y == yi).sum() / len(data.y)
        H_D += -Ck_D * np.log(Ck_D)
    return H_D


def claculate_H_D_A(data, A):
    H_D_A = 0
    for a in set(data.x.T[A]):
        H_D_A += (data.x.T[A] == a).sum() / len(data.y) * claculate_H_D(
            s_data(data.x[data.x.T[A] == a], data.y[data.x.T[A] == a]))
    return H_D_A


def claculate_max_g(data, index_list):
    max_index = index_list[0]
    max_g = 0
    H_D = claculate_H_D(data)
    for index in index_list:
        g_D_index = H_D - claculate_H_D_A(data, index)
        if max_g < g_D_index:
            max_index = index
            max_g = g_D_index
    if max_g!=0:
        print('max_g:', max_g)
    return max_index, max_g


def create_tree(data, index_list, _e=0.07):
    if len(Counter(data.y)) <= 1 or len(data.y) == 0 or len(index_list) < 1:
        return node(data, index_list)
    else:
        f_index, f_g = claculate_max_g(data, index_list)
        if f_g <= _e:
            return node(data, index_list)
        print(f_index, index_list)
        index_list.remove(f_index)
        branch = condition_node(
            f_index=f_index, index_list=index_list, data=data)
        branch.create_nodes(data)
        return branch


def predict(root, x):
    result = []
    for xi in x:
        locate = root
        while(1):
            try:
                locate = locate.branches[xi[locate.f_index]]
            except:
                result.append(locate.clas)
                break
    return result


def load_data():
    print('loading data...')
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
    X = X.T
    for i, j in zip([0, 2, 12], [15, 10000, 8]):
        row = X[i].astype(int)
        row = row - row % j
        X[i] = row.astype(str)
        print(X[i])
    X = X.T
    print('done!')
    number = 30000
    X_, y_ = X[:number], y[:number]
    print(X_.shape, y_.shape)
    data = s_data(X_, y_)
    index_list = (list(range(data.x.shape[1])))
    x = X[30000:35500]
    y = y[30000:35500]
    return data, index_list, x, y



def main():
    data, index_list, x, y = load_data()
    decision_tree = create_tree(data, index_list)
    pre_y = predict(decision_tree, x)
    print((np.array(pre_y) == y).sum() / len(y))


if __name__ == '__main__':
    main()
