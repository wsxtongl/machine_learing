import numpy as np
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,neighbors
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression,Ridge
import matplotlib.pyplot as plt

data = np.loadtxt(r'./红酒数据/wine.data',delimiter=',')
x = data[:,1:]
y = data[:,0].ravel()

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
#标准化处理
scaler = preprocessing.StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#创建模型
knn = neighbors.KNeighborsClassifier(n_neighbors=1)
#模型拟合
knn.fit(x_train,y_train)

y_predict = knn.predict(x_test)
print('准确率：',accuracy_score(y_test,y_predict))
print('真实值：',y_test[:10])
print('预测值：',y_predict[:10])
# k_x = range(1,50)
# k_score = []
# for i in k_x:
#     knn = neighbors.KNeighborsClassifier(n_neighbors=i)
#     scores = cross_val_score(knn,x,y,cv=5,scoring="accuracy")
#     k_score.append(scores.mean())
# plt.figure()
# plt.plot(k_x,k_score)
# plt.show()