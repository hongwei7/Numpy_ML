# Numpy_ML
使用Numpy搭建机器学习算法
## 已完成方法
  决策树：ID3、C4.5</br>
  逻辑回归</br>
  多层神经网络</br>
  KNN、基于kd树的KNN</br>
### 所使用数据：
UCI人口普查数据：该数据从美国1994年人口普查数据库抽取而来，可以用来预测居民收入是否超过50K。属性变量包含年龄，工种，学历，职业，人种等重要信息，14个属性变量中有7个类别型变量，是Classification/Regression模型训练的经典数据集。
### 具体表现
decision_tree_ID3_ex.py</br> test_accuracy:<strong>75.62%~81.92%</strong> </br>
decision_tree_ID3_cut.py</br> test_accuracy:<strong>81.253%</strong> used_time:4.0923s</br>
decision_tree_C4.5.py</br> test_accuracy:<strong>82.187%</strong> used_time:115.47s</br>
logistic_.py</br> test_accuracy:<strong>81.37%</strong> </br>
KNN_mat.py</br> test_accuracy:<strong>82.51%</strong> used_time:129.0s</br>
netrul_network(based on tensorflow).py</br> test_accuracy:<strong>84.786%</strong>
