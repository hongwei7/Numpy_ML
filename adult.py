#load
import numpy as np
file=open('adult.data')
r_X=[]
r_y=[]
for line in file.readlines():
    items=str(line).split(',')
    r_X.append(items[:-1])
    r_y.append(items[-1]==' >50K\n')
del r_X[32561],r_y[32561]
y=np.matrix(r_y).T.astype('int')
X=np.array(r_X)
def count(row):
    index=dict()
    number=0
    for i in row:
        if i not in index.keys():
            index[i]=number
            number+=1
    for key in index.keys():
        row[row==key]=index[key]
    return row.astype('float')
for num in [1,3,5,6,7,8,9,13]:
    count(X[:,num])
X=np.matrix(X.astype(float))
'done!'
from sklearn.linear_model import Lasso
clf=Lasso()
clf.fit(X,y)
pre_y=clf.predict(X)
print(pre_y)