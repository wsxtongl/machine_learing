import numpy as np
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,neighbors
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression,Ridge
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

data = np.loadtxt(r'./房价数据/housing.data')
x = data[:,:-1]
y = data[:,-1:].ravel()
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

lr = LinearRegression()
ridge = Ridge()

lr.fit(x_train,y_train)
ridge.fit(x_train,y_train)
y_predict1 = lr.predict(x_test)
y_predict2 = ridge.predict(x_test)
print('线性回归  r2得分：',r2_score(y_test,y_predict1))

print('岭回归  r2得分：',r2_score(y_test,y_predict2))

plt.xlim([0,50])
plt.plot( range(len(y_test)), y_test, 'r', label='真实值')
plt.plot( range(len(y_predict1)), y_predict1, 'b--', label='线性回归' )
plt.plot( range(len(y_predict2)), y_predict2, 'g--', label='岭回归' )
plt.title('sklearn: Linear Regression')
plt.ylabel('房价',size = 16)
plt.legend()
plt.show()

