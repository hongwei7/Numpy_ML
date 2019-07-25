import numpy as np
import matplotlib.pyplot as plt
import time
class perceptron():
    def __init__(self,eta=0.1):
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
        self.alpha=np.zeros((len(x)))
        self.w=np.zeros((x.shape[1]))
        self.b=0
        while not all_divided:
            num=np.random.randint(len(x))
            xi,yi=x[num],y[num]
            w_sum=y[0]*self.alpha[0]*x[0]
            for j,xj in enumerate(x[1:]):
                w_sum=w_sum+y[j+1]*self.alpha[j+1]*(x[j+1])
            g=yi*(w_sum.dot(xi)+self.b)
            if(g<=0):
                self.alpha[num]=self.alpha[num]+self.eta
                self.b=self.b+self.eta*yi
                w_sum=y[0]*self.alpha[0]*x[0]
                for j,xj in enumerate(x[1:]):
                    w_sum=w_sum+y[j+1]*self.alpha[j+1]*(x[j+1])
                self.w=w_sum
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