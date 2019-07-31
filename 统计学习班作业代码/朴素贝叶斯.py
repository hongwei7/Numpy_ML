import numpy as np
class Native_Bayes():
    def fit(self,X,y,lamb=0.2):
        self.X=X
        self.y=y
        self.lamb=lamb
    def cal_p_ci_x(self,ci,x):
        p_ci=(len(self.y[self.y==ci])+self.lamb)/(len(self.y)+self.lamb*
            len(set(self.y)))
        p_x_ci=1
        for index in range(len(x)):
            p_x_ci=p_x_ci*((sum((self.X[self.y==ci][:,index]==x[index])
                .astype('int'))+self.lamb)/(len(self.X[self.y[self.y==ci]])
            +self.lamb*len(set(self.X[:,index]))))
        return p_x_ci*p_ci
    def predict(self,x):
        p_ci_x={}
        for ci in set(self.y):
            p_ci_x[ci]=self.cal_p_ci_x(ci,x)
        print(p_ci_x)
        return sorted(p_ci_x.items(),key=lambda item:item[1])[-1][0]
def main():
    X=np.array([
        [1,'S'],[1,'M'],[1,'M'],[1,'S'],[1,'S'],[2,'S'],
        [2,'M'],[2,'M'],[2,'L'],[2,'L'],[3,'L'],[3,'M'],
        [3,'M'],[3,'L'],[3,'L']])
    y=np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
    x=['2','S']
    clf=Native_Bayes()
    clf.fit(X,y)
    pre_y=clf.predict(x)
    print(x,pre_y)
if __name__ == '__main__':
    main()