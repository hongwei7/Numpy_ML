import numpy as np
import tensorflow as tf
import random
class s_data(object):
    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y).T.reshape(len(y),1)

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
    X = X[0:60000]
    y = y[0:60000]
    X=X/(X.max(axis=0)-X.min(axis=0))
    X1=X1/(X1.max(axis=0)-X1.min(axis=0))
    print('done!')
    print('train_size:', X.shape, 'test_size:', X1.shape)
    data = s_data(X, y)
    test_data = s_data(X1, y1)
    index_list = (list(range(data.x.shape[1])))
    return data, test_data


def main():
    data,  test_data = load_data()
    print(data.x.shape,test_data.x.shape)
    epoch_size=5000
    first_layer_nodes=30
    second_layer_nodes=20
    x=tf.placeholder(tf.float32,shape=(None,14),name='input')
    y_=tf.placeholder(tf.float32,shape=(None,1),name='output')
    biases1=tf.Variable(tf.constant(0.1,shape=[epoch_size,first_layer_nodes]))
    biases2=tf.Variable(tf.constant(0.1,shape=[epoch_size,1]))
    w1=tf.Variable(tf.random_normal([14,first_layer_nodes],stddev=1))
    w2=tf.Variable(tf.random_normal([first_layer_nodes,second_layer_nodes],stddev=1))
    w3=tf.Variable(tf.random_normal([second_layer_nodes,1],stddev=1))
    a=tf.sigmoid(tf.matmul(x,w1)+biases1)
    b=tf.matmul(a,w2)
    y=tf.sigmoid(tf.matmul(b,w3)+biases2)
    cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0))+(1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
    learning_rate=0.01
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
    with tf.Session() as session:
        init_op=tf.global_variables_initializer()
        session.run(init_op)
        STEPS=100000
        for i in range(STEPS):
            rad=random.randint(0,data.x.shape[0]-epoch_size)
            feed_dict={x:data.x[rad:rad+epoch_size],y_:data.y[rad:rad+epoch_size]}
            session.run(train_step,feed_dict=feed_dict)
            if i%1000==0:
                total_cross_entropy=session.run(cross_entropy,feed_dict=feed_dict)
                print(i,'EPOCH,loss:',total_cross_entropy)
        print(STEPS,'EPOCH,loss:',session.run(cross_entropy,feed_dict=feed_dict))
        print('training_accuracy:',((session.run(y,feed_dict=feed_dict)>0.5)==data.y[rad:rad+epoch_size]).sum()/epoch_size*100,'%')
        pre_y=[]
        for i in range(0,test_data.x.shape[0]-test_data.x.shape[0]%epoch_size,epoch_size):
            xi=test_data.x[i:i+epoch_size]
            yi=test_data.y[i:i+epoch_size]
            t_xi=tf.Variable(xi,dtype=tf.float32)
            session.run(tf.variables_initializer([t_xi]))
            for bi in session.run(tf.sigmoid(tf.matmul(tf.matmul(tf.sigmoid(tf.matmul(t_xi,w1)+biases1),w2),w3)+biases2))>0.5:
                pre_y.append(bi[0])
        print('evaluating acccuracy:',(pre_y==test_data.y[:test_data.x.shape[0]-test_data.x.shape[0]%epoch_size].T).sum()/len(pre_y)*100,'%')
if __name__ == '__main__':
    main()
