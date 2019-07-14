from collections import Counter
import numpy as np
import time


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
    def __init__(self, f_index, index_list, data, divide_value, alpha):
        self.data = data
        self.alpha = alpha
        self.divide_value = divide_value
        self.f_index = f_index
        self.index_list = index_list
        self.clas = Counter(data.y).most_common()[0][0]
        # print('>=',divide_value,self.clas)#显示分类条件
        print(data.y[data.x.T[self.f_index] >= self.divide_value].sum(
        ) / len(data.y[data.x.T[self.f_index] >= self.divide_value]), 'divided\n')

    def create_nodes(self, data):
        self.branches = dict()
        self.branches[1] = create_tree(s_data(
            data.x[data.x.T[self.f_index] >= self.divide_value], data.y[data.x.T[self.f_index] >= self.divide_value]), self.index_list, alpha=self.alpha)
        self.branches[0] = create_tree(s_data(
            data.x[data.x.T[self.f_index] < self.divide_value], data.y[data.x.T[self.f_index] < self.divide_value]), self.index_list, alpha=self.alpha)


def claculate_H_D(data):
    H_D = 0
    for yi in set(data.y):
        Ck_D = (data.y == yi).sum() / len(data.y)
        H_D += -Ck_D * np.log(Ck_D)
    return H_D


def claculate_H_D_A(data, A, divide_value):
    H_D_A = (data.x.T[A] >= divide_value).sum() / len(data.y) * claculate_H_D(
        s_data(data.x[data.x.T[A] >= divide_value], data.y[data.x.T[A] >= divide_value]))
    H_D_A += (data.x.T[A] < divide_value).sum() / len(data.y) * claculate_H_D(
        s_data(data.x[data.x.T[A] < divide_value], data.y[data.x.T[A] < divide_value]))
    return H_D_A


def choose_divide_value(data, index, H_D):
    max_g = 0
    best_divide_value = data.x.T[0, 0]
    values = list(set(data.x.T[index]))
    values.sort()
    for x_, y_ in zip(values[:-1], values[1:]):
        ai = int((int(x_) + int(y_)) / 2)
        g = H_D - claculate_H_D_A(data, index, ai)
        if g >= max_g:
            max_g = g
            best_divide_value = ai
    return best_divide_value


def claculate_max_g(data, index_list, alpha):
    max_index = index_list[0]
    max_g = 0
    H_D = claculate_H_D(data)
    print('scanning ' + str(len(data.y)) + ' data...')
    for index in index_list:
        divide_value = choose_divide_value(data, index, H_D)
        H_D_A = claculate_H_D_A(data, index, divide_value)
        g_D_index = (H_D - H_D_A) / H_D  # +alpha
        if g_D_index >= max_g:
            max_index = index
            max_g = g_D_index
            best_divide_value = divide_value
    print(best_divide_value, 'best_divide_value', max_index, 'best_index')
    if max_g != 0:
        print('max_g:', max_g, 'best_divide_value=', best_divide_value)
    return max_index, max_g, best_divide_value


def create_tree(data, index_list, _e=0, alpha=0.2):
    if len(Counter(data.y)) <= 1 or len(data.y) <= 1 or len(index_list) < 1:
        return node(data, index_list)
    else:
        f_index, f_g, f_divide_value = claculate_max_g(data, index_list, alpha)
        if f_g <= _e:
            return node(data, index_list)
        print(f_index, index_list)
        index_list.remove(f_index)
        branch = condition_node(
            f_index=f_index, index_list=index_list, data=data, divide_value=f_divide_value, alpha=alpha)
        branch.create_nodes(data)
        return branch


def predict(root, x):
    result = []
    for xi in x:
        deep = 0
        locate = root
        while(1):
            deep += 1
            try:
                locate = locate.branches[int(
                    xi[locate.f_index] >= locate.divide_value)]
            except:
                result.append(locate.clas)
                break
    return result


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
    y = np.array(r_y).astype(float)
    X = np.array(r_X)
    y1 = np.array(r1_y).astype(float)
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
    X = X.astype(float)
    X1 = X1.astype(float)
    # 修改样本数量
    '''
    X = X[0:100000]
    y = y[0:100000]
    '''
    print('done!')
    print('train_size:', X.shape, 'test_size:', X1.shape)
    data = s_data(X, y)
    test_data = s_data(X1, y1)
    index_list = (list(range(data.x.shape[1])))
    return data, index_list, test_data


def main():
    data, index_list, test_data = load_data()
    t1 = time.time()
    decision_tree = create_tree(data, index_list, _e=0.1, alpha=0.)
    pre_y = predict(decision_tree, test_data.x)
    print('accuracy:' + str(100 * ((np.array(pre_y) ==
                                    test_data.y).sum() / len(pre_y)))[:6] + '%')
    print('error:' + str( (((np.array(pre_y) -
                                    test_data.y)**2).sum() / len(pre_y)))[:6] )
    t2 = time.time()
    print('used_time:' + str(t2 - t1)[:6] + 's')


if __name__ == '__main__':
    main()
