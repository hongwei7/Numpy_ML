import numpy as np
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
    X=X/X.max(axis=0)
    X=np.array(X)
    X1=X1/X1.max(axis=0)
    X1=np.array(X1)
    print('done!')
    print('train_size:', X.shape, 'test_size:', X1.shape)
    return X,y,X1,y1

def sigmoid(x):
    return 1/(1+np.exp(-x))
def random_gradAscant(X_mat,y):
    print(X_mat.shape,y.shape)
    print('start training!')
    alpha=0.01
    batch_size=50
    maxCycles=20000
    weights=np.random.rand(X_mat.shape[1],1)
    for k in range(maxCycles):
        random_num=np.random.randint(0,X_mat.shape[0]-batch_size)
        X_mat_i=X_mat[random_num:random_num+batch_size]
        y_i=np.matrix(y[random_num:random_num+batch_size])
        h=sigmoid(X_mat_i.dot(weights))
        loss=y_i.T-h
        weights=weights+alpha*X_mat_i.T.dot(loss)
        if k%1000==0 or (k+1)==maxCycles:
            print('after '+str(k+1)+' epoches,accuracy:'+str((predict(X_mat,weights).T==y).sum()/len(y)))
    print('done!')
    return weights
def predict(x,weights):
    return sigmoid((x).dot(weights))>=0.5
def main():
    print('preparing data...')
    X,y,X_test,y_test=load_data()
    weights=random_gradAscant(X,y)
    pre_y=predict(X_test,weights)
    print((pre_y.T==y_test).sum()/len(pre_y))
if __name__=='__main__':
    main()




