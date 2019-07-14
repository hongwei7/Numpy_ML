import pandas as pd
file=pd.read_csv('mnist_train.csv')
print(file.shape)
print(file)
mnist=dict()
print(file.columns)
mnist['data']=file
mnist['target']=file['label']
del mnist['data']['label']
mnist['data']=mnist['data'].values
mnist['target']=mnist['target'].values
X,y=mnist['data'],mnist['target']
X_train,y_train,X_test,y_test=X[:50000],y[:50000],X[50000:],y[50000:]
import numpy as np
shuffle_index=np.random.permutation(50000)
X_train,y_train=X_train[shuffle_index],y_train[shuffle_index]
from sklearn.neighbors import KNeighborsClassifier as kn
knn_clf=kn()
X_train=X_train[:1000]
y_train=y_train[:1000]
knn_clf.fit(X_train,y_train)
y_pred=knn_clf.predict(X_train)
from sklearn.model_selection import GridSearchCV as grid
param=[{'weights':['distance'],'n_neighbors':[2,3,4],'p':[4,5,6]}]
kn_clf=kn()
grid_clf=grid(kn_clf,param,cv=2,scoring='neg_mean_squared_error')
print('start!')
import time
t1=time.time()
grid_clf.fit(X_train,y_train)
t2=time.time()
from matplotlib import pyplot as plt
print(grid_clf.best_estimator_)
print(t2-t1,'seconds')
y_test_pred=grid_clf.best_estimator_.predict(X_test)
from sklearn.metrics import confusion_matrix
cm_test=confusion_matrix(y_test,y_test_pred)
plt.imshow(cm_test,cmap=plt.cm.gray)
plt.show()
print(cm_test)
