from collections import Counter
import numpy as np
import time

class _data(object):#自定义数据集类型
    def __init__(self, x, y):
        self.x = x
        self.y = y


class node(object):#单节点
    def __init__(self, data, index_list):
        self.data = data
        self.clas = Counter(data.y).most_common()[0][0]
        self.index_list = index_list


class condition_node(object):#非单节点
    def __init__(self, f_index, index_list, data, alpha):
        self.alpha = alpha
        self.data = data
        self.f_index = f_index
        self.index_list = index_list
        self.clas = Counter(data.y).most_common()[0][0]

    def create_nodes(self, data):
        self.branches = dict()
        for f in set(data.x.T[self.f_index]):
            self.branches[f] = create_tree(_data(
                data.x[data.x.T[self.f_index] == f], 
                data.y[data.x.T[self.f_index] == f]), 
            self.index_list, alpha=self.alpha)


def claculate_H_D(data):#计算信息熵
    H_D = 0
    for yi in set(data.y):
        Ck_D = (data.y == yi).sum() / len(data.y)
        H_D += -Ck_D * np.log(Ck_D)
    return H_D


def claculate_H_D_A(data, A):#计算条件熵
    H_D_A = 0
    for a in set(data.x.T[A]):
        H_D_A += (data.x.T[A] == a).sum() / len(data.y) * claculate_H_D(
            _data(data.x[data.x.T[A] == a], data.y[data.x.T[A] == a]))
    return H_D_A


def claculate_max_g(data, index_list, alpha): #alpha为预剪枝参数
    max_index = index_list[0]
    max_g = 0
    H_D = claculate_H_D(data)
    for index in index_list:
        g_D_index = (H_D - claculate_H_D_A(data, index)) 
        + alpha * (len(Counter(data.x.T[index]).keys()) - 1) #预剪枝
        if max_g < g_D_index:
            max_index = index
            max_g = g_D_index
    if max_g != 0:
        pass
        #print('max_g:', max_g) #显示信息增益具体值
    return max_index, max_g


def create_tree(data, index_list, alpha,_e=0.07):
    if len(Counter(data.y)) <= 1 or len(data.y) == 0 or len(index_list) < 1:
        return node(data, index_list)#返回单节点树
    else:
        f_index, f_g = claculate_max_g(data, index_list,alpha)
        if f_g <= _e:
            return node(data, index_list)#返回单节点树
        #print(f_index, index_list) #显示分类条件
        index_list.remove(f_index)
        branch = condition_node(
            f_index=f_index, index_list=index_list, data=data,alpha=alpha)
        branch.create_nodes(data)
        return branch

def predict(root, x):
    result = []
    for xi in x:
        locate = root #locate作为指针
        while(1):
            try:
                locate = locate.branches[xi[locate.f_index]]
            except:
                result.append(locate.clas)
                break
    return result


def load_data(file_name='adult.data', condition=' >50K\n'):
    print('loading data... ' + file_name)
    file = open(file_name)
    r_X,r_y= [],[]
    for line in file.readlines():
        items = str(line).split(',')
        r_X.append(items[:-1])
        r_y.append(items[-1] == condition)
    del r_X[-1], r_y[-1]
    X,y = np.array(r_X).T,np.array(r_y).astype('int')
    for i in [0,2,10,11,12]:  # 连续变量离散化
        row=X[i].astype(int)
        X[i] = (row>np.median(row)).astype(int).astype(str)
    X = X.T
    print(file_name + ' size:', X.shape)
    data = _data(X, y)
    index_list = list(range(data.x.shape[1]))
    return data, index_list
 

def main():
    data, index_list = load_data()
    test_data, _index_list = load_data('adult.test', condition=' >50K.\n')
    t1=time.time()
    decision_tree = create_tree(data, index_list, 0.001,0.0007)
    pre_y = predict(decision_tree, data.x)
    print('train accuracy:',(np.array(pre_y) == data.y).sum() / len(pre_y))
    testpre_y = predict(decision_tree, test_data.x)
    print('test accuracy:',(np.array(testpre_y) == test_data.y).sum() / len(testpre_y))
    t2=time.time()
    print('used_time:'+str(t2-t1)[:6]+'s')

if __name__ == '__main__':
    main()
