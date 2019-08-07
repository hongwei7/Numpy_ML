import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
class logistic(object):
    def __init__(self,X,y):
        self.X=np.c_[X,np.ones([len(X),1])]
        self.y=y
        self.w=np.ones([X.shape[1]+1])
    def train(self,epoches=1000,eta=0.1):
        for epo in range(epoches):
            diff=np.zeros(self.w.shape)
            for xi,yi in zip(self.X,self.y):
                diff=diff+(yi-np.exp(self.w.dot(xi))/(1+np.exp(self.w.dot(xi))))*xi
            self.w=self.w+eta*diff

    def predict(self,x):
        x=np.r_[x,np.ones([1])]
        return int((np.exp(self.w.dot(x))/(1+np.exp(self.w.dot(x))))>0.5)

def main():
    X=np.array([[3,3,3],[4,3,2],[2,1,2],[1,1,1],[-1,0,1],[2,-2,1]])
    y=np.array([1,1,1,0,0,0])
    x=np.array([1,2,-2])
    clf=logistic(X,y)
    clf.train()
    print(x,'预测值为:',clf.predict(x))
    ax = plt.subplot(projection='3d') 
    ax.scatter(X.T[0,:3],X.T[1,:3],X.T[2,:3],c='r')
    ax.scatter(X.T[0,3:],X.T[1,3:],X.T[2,3:],c='b')
    ax.scatter(1,2,-2,c='g')
    plt.show()
if __name__ == '__main__':
    main()