import numpy as np
import matplotlib.pyplot as plt
import time
class perceptron():
    def __init__(self,eta=1):
        self.eta=eta
    def sign(self,f):
        if f>=0:
            return 1
        else:
            return -1
    def predict(self,xi):
        return(self.sign(self.w.dot(xi)+self.b))
    def train(self,x,y):
        all_divided=False
        self.w=np.zeros((x.shape[1]))
        self.b=0
        while not all_divided:
            num=np.random.randint(len(x))
            xi,yi=x[num],y[num]
            pre_yi=self.predict(xi)
            if(pre_yi!=yi):
                self.w=self.w+self.eta*xi*yi
                self.b=self.b+self.eta*yi
            all_divided=True
            for i,xi in enumerate(x):
                if self.predict(xi)!=y[i]:
                    all_divided=False
                    break
def main():
    x=np.array([[3,3],[4,3],[1,1]])
    y=np.array([1,1,-1])
    clf=perceptron()
    t1=time.time()
    clf.train(x,y)
    print('used_time:',time.time()-t1)   
    for i,xi in enumerate(x):
        print(xi,clf.predict(xi),y[i])
    plt.plot([3,4],[3,3],'r.')
    plt.plot(1,1,'g.')
    plt.plot([0,4],[(-clf.b-clf.w[0]*0)/clf.w[1],
        (-clf.b-clf.w[0]*4)/clf.w[1]],'b')
    plt.show()
if __name__ == '__main__':
    main()