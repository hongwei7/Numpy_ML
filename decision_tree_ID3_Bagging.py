from collections import Counter
import numpy as np
import time

class _data(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
class tree(object):
    def __init__(self, data,index_list,alpha,e):
        self.clf = create_tree(data,index_list,alpha,e)
        

class node(object):
    def __init__(self, data, index_list):
        self.data = data
        self.clas = Counter(data.y).most_common()[0][0]
        self.index_list = index_list


class condition_node(object):
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
                data.x[data.x.T[self.f_index] == f], data.y[data.x.T[self.f_index] == f]), self.index_list, alpha=self.alpha)


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
            _data(data.x[data.x.T[A] == a], data.y[data.x.T[A] == a]))
    return H_D_A


def claculate_max_g(data, index_list, alpha):
    max_index = index_list[0]
    max_g = 0
    H_D = claculate_H_D(data)
    for index in index_list:
        g_D_index = (H_D - claculate_H_D_A(data, index)) + alpha * (len(Counter(data.x.T[index]).keys()) - 1)
        if max_g < g_D_index:
            max_index = index
            max_g = g_D_index
    if max_g != 0:
        1#print('max_g:', max_g)
    return max_index, max_g


def create_tree(data, index_list, alpha,_e=0.07):
    if len(Counter(data.y)) <= 1 or len(data.y) == 0 or len(index_list) < 1:
        return node(data, index_list)
    else:
        f_index, f_g = claculate_max_g(data, index_list,alpha)
        if f_g <= _e:
            return node(data, index_list)
        #print(f_index, index_list)
        index_list.remove(f_index)
        branch = condition_node(
            f_index=f_index, index_list=index_list, data=data,alpha=alpha)
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

class Bagging_tree(object):
    def __init__(self, data,index_list,num,alpha=0,e=0):
        self.data = data
        self.index_list=index_list
        self.num=num
        self.clf_list=[]
        self.alpha=alpha
        self.e=e
        self.train()
    def train(self):
        permutation = np.random.permutation(self.data.x.shape[0])
        self.data.x=self.data.x[permutation,:]
        self.data.y=self.data.y[permutation]
        for i in range(self.num):
            index=[]
            for j in self.index_list:
                index.append(j)
            front,back=len(self.data.x)//self.num*i,len(self.data.x)//self.num*(i+1)
            data_t=_data(self.data.x[front:back],self.data.y[front:back])
            self.clf_list.append(tree(data_t,index,self.alpha,self.e))
            pre_yt=predict(self.clf_list[-1].clf,data_t.x)
            print('single predict:',(np.array(pre_yt) == data_t.y).sum() / len(pre_yt))
    def predict(self,x):
        #print(self.clf_list)
        result=np.array(np.zeros([len(x),1])).reshape(len(x))
        for clf in self.clf_list:
            result=result+np.array(predict(clf.clf,x))/self.num
        return np.array(result)>=0.5



def load_data(file_name='adult.data', condition=' >50K\n'):
    print('loading data... ' + file_name)
    file = open(file_name)
    r_X = []
    r_y = []
    for line in file.readlines():
        items = str(line).split(',')
        r_X.append(items[:-1])
        r_y.append(items[-1] == condition)
    del r_X[-1], r_y[-1]
    y = np.array(r_y).astype('int')
    X = np.array(r_X)
    X = X.T
    for i, j in zip([0, 2, 12], [10, 10000, 5]):  # 连续变量离散化
        row = X[i].astype(int)
        row = row - row % j
        X[i] = row.astype(str)
    X = X.T

    print('done!')
    print(file_name + ' size:', X.shape)
    data = _data(X, y)
    index_list = (list(range(data.x.shape[1])))
    return data, index_list


        
def main():
    data, index_list = load_data()
    test_data, _index_list = load_data('adult.test', condition=' >50K.\n')
    t1=time.time()
    bagging_tree=Bagging_tree(data,index_list,5,0.0,0.01)
    pre_y=bagging_tree.predict(data.x)
    print('bagging train accuracy:',(np.array(pre_y) == data.y).sum() / len(pre_y))
    test_pre_y=bagging_tree.predict(test_data.x)
    print('bagging test accuracy:',(np.array(test_pre_y) == test_data.y).sum() / len(test_pre_y))
    t2=time.time()
    print('used_time:'+str(t2-t1)[:6]+'s')
    # print(data.y.sum(),test_data.y.sum())
if __name__ == '__main__':
    main()
