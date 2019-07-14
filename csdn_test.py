
#! /usr/bin/env python2.7
# coding=utf-8
 
import numpy as np
import pandas as pd
from sklearn import preprocessing
#from sklearn.cross_validation import train_test_split
#import seaborn as sns
import matplotlib.pyplot as plt
#导入高斯库
from sklearn.naive_bayes import GaussianNB
#线性回归建模
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
 
file1 = "/***/data/salarytrain.dat"
file2 = '/***/data/salarytest.dat'
train = pd.DataFrame(pd.read_csv(file1, na_values=' ?'))
test = pd.DataFrame(pd.read_csv(file2, na_values=' ?'))
# print train.shape
 
#此处，不处理缺失值相比简单处理缺失值，在使用朴素贝叶斯方法预测时准确率还要更高，故目前不处理缺失值。后续待定
# train = train.bfill(axis=0)
# test = test.bfill(axis=0)
# print train
train.columns = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Educationnum',
                 'Maritalstatus', 'Occupation', 'Relationship', 'Race', 'Sex',
                 'Capitalgain', 'Capitalloss', 'Hoursperweek', 'Nativecountry', 'Income']
#此处查看的是原始数据缺失值的总数
# print train.isnull().sum()
print(train.index)
test.columns = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Educationnum',
                 'Maritalstatus', 'Occupation', 'Relationship', 'Race', 'Sex',
                 'Capitalgain', 'Capitalloss', 'Hoursperweek', 'Nativecountry']
 
#数据预处理：string类型转int类型，归一化
#1：用LabelEncoder处理非数值型特征值
leEncoder = preprocessing.LabelEncoder()
leCols = [train.Workclass, train.Education, train.Maritalstatus,
          train.Occupation, train.Relationship, train.Race,
          train.Sex, train.Nativecountry, train.Income]
workclass = leEncoder.fit_transform(leCols[0])
education = leEncoder.fit_transform(leCols[1])
maritalstatus = leEncoder.fit_transform(leCols[2])
occupation = leEncoder.fit_transform(leCols[3])
relationship = leEncoder.fit_transform(leCols[4])
race = leEncoder.fit_transform(leCols[5])
sex = leEncoder.fit_transform(leCols[6])
nativecountry = leEncoder.fit_transform(leCols[7])
income = leEncoder.fit_transform(leCols[8])
# print type(workclass)
# print type(train.Age)
 
leCols2 = [test.Workclass, test.Education, test.Maritalstatus,
           test.Occupation, test.Relationship, test.Race,
           test.Sex, test.Nativecountry]
workclass2 = leEncoder.fit_transform(leCols2[0])
education2 = leEncoder.fit_transform(leCols2[1])
maritalstatus2 = leEncoder.fit_transform(leCols2[2])
occupation2 = leEncoder.fit_transform(leCols2[3])
relationship2 = leEncoder.fit_transform(leCols2[4])
race2 = leEncoder.fit_transform(leCols2[5])
sex2 = leEncoder.fit_transform(leCols2[6])
nativecountry2 = leEncoder.fit_transform(leCols2[7])
# print workclass[0:50]
 
#2.对连续数值数据进行归一化处理，处理第3和11列数据
newFnlwgt = (train['Fnlwgt'] - train['Fnlwgt'].min()) / (train['Fnlwgt'].max() - train['Fnlwgt'].min())
newCapitalgain = (train['Capitalgain'] - train['Capitalgain'].min()) / (train['Capitalgain'].max() - train['Capitalgain'].min())
newCapitalloss = (train['Capitalloss'] - train['Capitalloss'].min()) / (train['Capitalloss'].max() - train['Capitalloss'].min())
# print type(newCapitalgain)
 
newFnlwgt2 = (test['Fnlwgt']- test['Fnlwgt'].min()) / (test['Fnlwgt'].max() - train['Fnlwgt'].min())
newCapitalgain2 = (test['Capitalgain'] - test['Capitalgain'].min()) / (train['Capitalgain'].max() - train['Capitalgain'].min())
newCapitalloss2 = (test['Capitalloss'] - test['Capitalloss'].min()) / (train['Capitalloss'].max() - train['Capitalloss'].min())
 
#3.把数据集重新组合在一起
train_data = pd.concat([train.Age, train.Educationnum, newFnlwgt, newCapitalgain,
                        newCapitalloss, train.Hoursperweek], axis=1)
train_data['Workclass'] = workclass
train_data['Education'] = education
train_data['Maritalstatus'] = maritalstatus
train_data['Occupation'] = occupation
train_data['Relationship'] = relationship
train_data['Race'] = race
train_data['Sex'] = sex
train_data['Nativecountry'] = nativecountry
train_data['Income'] = income
 
test_data = pd.concat([test.Age, test.Educationnum, newFnlwgt2, newCapitalgain2,
                       newCapitalloss2, test.Hoursperweek], axis=1)
test_data['Workclass'] = workclass2
test_data['Education'] = education2
test_data['Maritalstatus'] = maritalstatus2
test_data['Occupation'] = occupation2
test_data['Relationship'] = relationship2
test_data['Race'] = race2
test_data['Sex'] = sex2
test_data['Nativecountry'] = nativecountry2
 
# print train_data.head(10)
# print test_data.head(10)
 
sns.set(context='paper', font='monospace')
sns.set(style='white')
f, ax = plt.subplots(figsize=(10, 15))
sns.heatmap(train_data, ax=ax, vmax=.9, square=True)
ax.set_xticklabels(train_data.index, size=15)
ax.set_yticklabels(train_data.columns[::-1], size=15)
ax.set_title('train feature', fontsize=20)
plt.show()
 
 
#下面根据各个参数独立查看薪水状况，找出具有明显分布的特征
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里分列几个小图
# train_data.Income.value_counts().plot(kind='bar')  # 柱状图
# plt.title(u"薪水情况（1为超过5k)")
# plt.ylabel(u"人数")
#
# plt.subplot2grid((2, 3), (0, 1), colspan=2)
# train_data.Income[train_data.Education == 7].plot(kind='kde')
# train_data.Income[train_data.Education == 8].plot(kind='kde')  # 密度图
# train_data.Income[train_data.Education == 9].plot(kind='kde')
# train_data.Income[train_data.Education == 11].plot(kind='kde')
# train_data.Income[train_data.Education == 12].plot(kind='kde')  # 密度图
# train_data.Income[train_data.Education == 15].plot(kind='kde')
# plt.xlabel("Education")  # plots an axis lable
# plt.ylabel(u"密度")
# plt.title(u"各教育水平的薪水分布")
# plt.legend(('7', '8', '9', '11', '12', '15'), loc='best')
#
# plt.subplot2grid((2, 3), (1, 0))
# plt.scatter(train_data.Income, train_data.Age)  # 为散点图传入数据
# plt.ylabel(u"年龄")
# plt.grid(b=True, which='major', axis='y')
# plt.title(u"按年龄看薪水分布")
#
# plt.subplot2grid((2, 3), (1, 1))
# plt.scatter(train_data.Income, train_data.Nativecountry)  # 为散点图传入数据
# plt.ylabel(u"原始国籍")
# plt.grid(b=True, which='major', axis='y')
# plt.title(u"按原始国籍看薪水分布")
#
# #通过下图可以看出，薪水高于5k中，男性比女性多，低于5k中，女性居多，可以较好的区分，故可给性别属性较大的权重
# plt.subplot2grid((2, 3), (1, 2), colspan=2)
# train_data.Income[train_data.Sex == 1].plot(kind='kde')  # 密度图
# train_data.Income[train_data.Sex == 0].plot(kind='kde')
# plt.xlabel("sex")  # plots an axis lable
# plt.ylabel(u"密度")
# plt.title(u"各性别的薪水分布")
# plt.legend(('female', 'male'), loc='best')  # sets our legend for our graph.
 
# plt.show()
 
#分割训练集和验证集
training, validation = train_test_split(train_data, train_size=0.7)
features = ['Age', 'Workclass', 'Fnlwgt', 'Education', 'Educationnum',
            'Maritalstatus', 'Occupation', 'Relationship', 'Race', 'Sex',
            'Capitalgain', 'Capitalloss', 'Hoursperweek', 'Nativecountry']
# print validation.shape
 
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.naive_bayes import BernoulliNB
import time
 
#建立伯努利模型朴素贝叶斯，计算log_loss
model = BernoulliNB()
nbstart = time.time()
model.fit(training[features], training['Income'])
nbCostTime = time.time() - nbstart
# predicted = np.array(model.predict_proba(validation[features]))
predicted = np.array((model.predict(validation[features])))
# print predicted
print('伯努利朴素贝叶斯建模耗时 %f 秒' % nbCostTime)
print("朴素贝叶斯log损失为 %f" % (log_loss(validation['Income'], predicted)))
print("朴素贝叶斯的准确率为 %f" % (precision_score(validation['Income'], predicted)))
 
#建立高斯朴素贝叶斯模型，计算准确率
model2 = GaussianNB()
nbstart2 = time.time()
predicted2 = model2.fit(training[features], training['Income'])
nbCostTime2 = time.time() - nbstart2
predicted2 = np.array((model2.predict(validation[features])))
print('高斯朴素贝叶斯建模耗时 %f 秒' % nbCostTime2)
print("朴素贝叶斯log损失为 %f" % (log_loss(validation['Income'], predicted2)))
print("朴素贝叶斯的准确率为 %f" % (precision_score(validation['Income'], predicted2)))
 
# 线性回归建模
# 线性回归准确率太低。。。原因是：线性回归并不适合分类，适合的是拟合
lr = LinearRegression()
#先把train分为训练集和测试集，再把训练样本分成三份，用来交叉验证
kf = KFold(train_data.shape[0], n_folds=3, random_state=1)
predictions = []
for tra, test in kf:
    tra_predictors = (train_data[features].iloc[tra, :])
    tra_target = train_data['Income'].iloc[tra]
    lr.fit(tra_predictors, tra_target)
 
    test_predictions = lr.predict(train_data[features].iloc[test, :])
    predictions.append(test_predictions)
 
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > 0.5] = 1
predictions[predictions < 0.5] = 0
 
accuracy = sum(predictions[predictions == train_data['Income']]) / len(predictions)
print(accuracy)
 
# 逻辑回归
from sklearn.linear_model import LogisticRegression #逻辑回归
from sklearn import cross_validation
nbstart4 = time.time()
alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg, train_data[features],train_data['Income'], cv=3)
nbCostTime4 = time.time() - nbstart4
print('逻辑回归建模耗时 %f 秒' % nbCostTime4)
print("逻辑回归预测模型的准确率为 %f" % scores.mean())
 
#用随机森林来拟合
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
 
nbstart3 = time.time()
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=2)
kf = KFold(train_data.shape[0], n_folds=3, random_state=1)
scores2 = cross_validation.cross_val_score(alg, train_data[features], train_data['Income'], cv=kf)
nbCostTime3 = time.time() - nbstart3
print('随机森林建模耗时 %f 秒' % nbCostTime3)
print("回归森林预测模型的准确率为 %f" % scores2.mean())
