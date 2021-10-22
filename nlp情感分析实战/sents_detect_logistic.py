"""属于nlp 文本分类的任务
数据集是 tag，text 的形式"""

import  numpy as np
import  pandas as pd


# 加载data
data=pd.read_csv("ISEAR.csv",header=None)
print(data.head())
print(data.describe())

#获取样本  和目标值
from sklearn.model_selection import train_test_split
labels=data[0].values.tolist()
sents=data[1].values.tolist()
X_train, X_test, y_train, y_test=train_test_split(sents,labels,test_size=0.2,random_state=42)


#tf-idf进行表征
from  sklearn.feature_extraction.text import  TfidfVectorizer
vectorizer=TfidfVectorizer()
#利用训练集的文本的进行tf idf 的统计学习   然后把训练集和测试集用数字化的形式表示
X_train=vectorizer.fit_transform(X_train)    #训练t'fidf  并进行数据转换
X_test=vectorizer.transform(X_test)    #根据fit的统计结果 只进行数据特征化
"""还可以用其他的方法进行特征提取结合一起表示 ：如加入词性的特征、n_gram的信息，单词本身的信息 等 特征共工程"""



#构建逻辑回归模型和网格交叉搜索
from  sklearn.linear_model import LogisticRegression
from  sklearn.model_selection import GridSearchCV


#需要搜索的参数的范围  参数越多，计算量指数级别上涨
parameters = {'C':[0.00001, 0.0001, 0.001, 0.005,0.01,0.05, 0.1, 0.5,1,2,5,10]}
model=LogisticRegression()
#指定训练数据  和损失分数的数据
model.fit(X_train,y_train).score(X_test,y_test)

#网格交叉搜索
clf=GridSearchCV(model,parameters,cv=10)
#开始训练
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
print(clf.best_params_)   #5
print(clf.best_score_) #{'C': 2}
#0.5848367339041797



#混淆矩阵   一个多分类任务的评价指标
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, clf.predict(X_test))
"""
混淆矩阵是表示精度评价的一种标准格式   主要出现在多分类任务中
每一列代表了预测类别，每一列的总数表示预测为该类别的数据的数目；
每一行代表了数据的真实归属类别，每一行的数据总数表示该类别的数据实例的数目。
每一列中的数值表示真实数据被预测为该类的数目   对角线越大说明模型效果越好 ，预测的越精准
array([[101,  35,  17,  26,  11,  18,  19],
       [ 23, 122,  14,  18,   8,   5,  14],
       [ 12,   7, 141,  12,   8,  11,   9],
       [ 25,  10,  11, 110,  12,  17,  24],
       [  5,   9,   8,   9, 180,  13,   9],
       [ 20,  16,  12,   6,  18, 126,   7],
       [ 23,  23,  17,  33,  18,   8, 104]])
​"""