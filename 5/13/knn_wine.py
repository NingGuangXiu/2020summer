import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#加载数据
df = pd.read_csv("wine.csv")
df.head()
print("该数据集共有 {} 行 {} 列".format(df.shape[0],df.shape[1]))
#特征
X = np.array(df[df.columns[:13]])
#分类标签
y = np.array(df.Class)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

#初始化
k = 3
clf = KNeighborsClassifier(k)
#使用training set训练模型
clf.fit(X_train, y_train)
# training set正确率
print("训练集正确率：{}%".format(round(clf.score(X_train, y_train)*100,2)))