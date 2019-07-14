import pandas as pd
import numpy as np
print('preparing data...')
train = pd.read_csv('happiness_train_complete1.csv')
train = train.loc[train['happiness'] != -8]
del_list = ['id', 'survey_time', 'edu_other', 'property_other','happiness', 'invest_other']
y = train.happiness.values
for index in del_list:
    del train[index]
for i in train.columns:
    train[i].astype(float)
    train[i][train[i] < 0] = train[i].median()
    train[i].fillna(train[i].median(), inplace=True)
print('done!')
X = np.matrix(train.values)
X=X/X.max(axis=1)
X=np.array(X)

def fit(X,y,theta=None):
    eta=0.1
    sort_number=5    #sortnumber代表类别个数 以及类别标签
    if theta==None:
        theta=np.random.rand(X.shape[1],sort_number)
        old_theta=np.zeros((X.shape[1],sort_number))
    for i in range(5):
        old_theta=theta
        for k in range(sort_number):
            yk=y==k+1
            yk=yk.astype(int)
            s=(old_theta.T).dot(X.T)
            sk=s[k,:]
            s_sum=np.exp(s[0,:])
            for j in range(sort_number):
                s_sum=s_sum+np.exp(s[j,:])
            pk=np.exp(sk)/s_sum
            delta=1/len(X)*((pk-yk)).dot(X)
            theta[:,k]=old_theta[:,k]-eta*delta
    return theta
def predict(x,theta):
    pro=np.exp(theta.T.dot(x.T))/sum(np.exp(theta.T.dot(x.T)))
    return np.argmax(pro,axis=0)
def early_stopping_predict(X,y,x):
    from sklearn.model_selection import train_test_split
    x_train,x_val,y_train,y_val=train_test_split(X,y,test_size=0.2)
    best_theta=None
    best_epoch=None
    min_val_error=float('inf')
    for epoch in range(500):
        theta=fit(x_train,y_train)
        train_mean_error=(((predict(x_train,theta)-y_train)**2).sum()/len(x_train))**0.5
        val_mean_error=(((predict(x_val,theta)-y_val)**2).sum()/len(x_val))**0.5
        if val_mean_error<min_val_error:
            best_theta=theta
            best_epoch=epoch
            min_val_error=val_mean_error
            best_train_error=train_mean_error
    print('train_error:',best_train_error,'val_error:',min_val_error,'best_epoch:',best_epoch)
    return predict(x,best_theta)
pre_y=early_stopping_predict(X,y,X)
