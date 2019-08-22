import numpy as np

class single_tree(object):
    def __init__(self):
        return
    def train(self,X,y,alpha,X0):
        self.y,self.alpha=y,alpha
        min_loss=float('inf')
        best_f,best_x_val,best_y_val=0,0,0
        for index in range(X.shape[1]):
            for x_val in set(X[:,index]):
                for yi in set(y):
                    self.f,self.x_val,self.y_val=index,x_val,yi
                    loss=0
                    for i,xi in enumerate(X):
                        loss=loss+(self.predict(xi)!=y[i])*alpha[i]
                    if loss<=min_loss:
                        min_loss=loss
                        best_f,best_x_val,best_y_val=index,x_val,yi
        #print('min_loss',min_loss)
        self.f,self.x_val,self.y_val=best_f,best_x_val,best_y_val

    def predict(self,xi):
        if xi[self.f]==self.x_val:
            return self.y_val
        else:
            for yi in set(self.y):
                if yi!=self.y_val:
                    return yi


class Adaboost(object):
    def __init__(self,m=7):
        self.m=m
        self.clf_list=[]
    def train(self,X,y):
        self.X,self.y=X,y
        Xm=X
        alpha=np.ones(X.shape[0])/len(X)
        for epoch in range(self.m):
            clf=single_tree()
            clf.train(Xm,y,alpha,self.X)
            em=(self.predict(X)!=y).sum()/len(X)
            if em==0:
                #print('min_loss',0)
                break
            km=0.5*np.log((1-em)/em)
            self.clf_list.append([km,clf])
            alpha=self.update_alpha(alpha,km)

    def update_alpha(self,alpha,km):
        Z=0
        pre_y=self.predict(self.X)
        for i,ai in enumerate(alpha):
            alpha[i]=ai*np.exp(-km*self.y[i]*pre_y[i])
            Z=Z+ai*np.exp(-km*self.y[i]*pre_y[i])
        return alpha/Z
    def predict(self,X):
        result=[]
        for xi in X:
            yi=0
            for k,clf in self.clf_list:
                yi=yi+k*clf.predict(xi)
            if yi>0:
                yi=1
            else:
                yi=-1
            result.append(yi)
        return np.array(result)
def main():
    X=np.array([[0,1,3],[0,3,1],[1,2,2],[1,1,3],
        [1,2,3],[0,1,2],[1,1,2],[1,1,1],[1,3,1],[0,2,1]])
    y=np.array([-1,-1,-1,-1,-1,-1,1,1,-1,-1]).T
    clf=Adaboost(20)
    clf.train(X,y)
    print(y)
    print(clf.predict(X))
if __name__ == '__main__':
    main()