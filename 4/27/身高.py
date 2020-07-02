import pandas as pd
from io import StringIO
from sklearn import linear_model
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import DataFrame,Series
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#
data={'母亲':[154,157,158,159,160,161,162,163],'女儿':[155,156,159,162,161,164,165,166]}
data1=DataFrame(data)
x=data1.母亲
y=data1.女儿
X_train,X_test,Y_train,Y_test = train_test_split(x,y,train_size=.9)

print("原始数据特征:",x.shape,      ",训练数据特征:",X_train.shape,      ",测试数据特征:",X_test.shape)
print("原始数据标签:",x.shape,      ",训练数据标签:",Y_train.shape,      ",测试数据标签:",Y_test.shape)

model = LinearRegression()
X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)
model.fit(X_train,Y_train)
a  = model.intercept_
b = model.coef_
print("最佳拟合线:截距",a,",回归系数：",b)

y_train_pred = model.predict(X_train)
plt.plot(X_train, y_train_pred, color='yellow', linewidth=3, label="Regression line")
#测试数据散点图
plt.scatter(X_train, Y_train, color="blue", label="train data")
plt.scatter(X_test, Y_test, color='red', label="test data")
#添加图标标签
plt.legend(loc=2)
plt.xlabel("Hours")
plt.ylabel("Score")
plt.show()

X = [[167]]
print ('母亲身高为167时，女儿的身高预测为：',model.predict(X))