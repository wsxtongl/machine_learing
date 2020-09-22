import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import r2_score,accuracy_score,explained_variance_score,mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor


plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

mon_dict = {
    '1':'jan',
    '2':'feb',
    '3':'mar',
    '4':'apr',
    '5':'may',
    '6':'jun',
    '7':'jul',
    '8':'aug',
    '9':'sep',
    '10':'oct',
    '11':'nov',
    '12':'dec'
}
week_dict = {
    '1':'mon',
    '2':'tue',
    '3':'wed',
    '4':'thu',
    '5':'fri',
    '6':'sat',
    '7':'sun',
}

fire_data_x = []
fire_data_y = []
with open(r'./森林火灾/forestfires.csv') as file:

    file = file.readlines()[1:]
    for line in file:
        str = line.strip().split(',')
        for key, value in mon_dict.items():
            if str[2] == value:
                str[2] = key
        for key, value in week_dict.items():
            if str[3] == value:
                str[3] = key
        fire_data_x.append(str[:-1])
        fire_data_y.append(str[-1])
x = np.float64(np.stack(fire_data_x))
y1 = np.float64(np.stack(fire_data_y))
y = np.log1p(y1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state = 1)
scaler = preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
epsilon = np.arange(0.1,1.5,0.2)
C= np.arange(1,500,10)
gamma = np.arange(0.001,0.01,0.002)
parameters = {'epsilon':epsilon,'C':C,'gamma':gamma}
grid_svr = GridSearchCV(estimator = SVR(),param_grid =parameters,
                                        scoring='neg_mean_squared_error',cv=5,verbose =1, n_jobs=-1)

models=[LinearRegression(),KNeighborsRegressor(),grid_svr,Ridge(),
        Lasso(),MLPRegressor(alpha=20),
        DecisionTreeRegressor(),ExtraTreeRegressor(),
        RandomForestRegressor(),AdaBoostRegressor(),GradientBoostingRegressor(),BaggingRegressor()]
models_str=['LinearRegression','KNNRegressor','SVR','Ridge','Lasso',
            'MLPRegressor','DecisionTree','ExtraTree',
'RandomForest','AdaBoost','GradientBoost','Bagging']
score_=[]

# svr = SVR(kernel='rbf',C=50)
# lr= LinearRegression()
#lr = Ridge()
#sigma = 5
#clf = KernelRidge(alpha=1.0, kernel='rbf',gamma=sigma ** -2)
#clf.fit(x_train,y_train)
y_pre = []
for name,model in zip(models_str,models):
    print('开始训练模型：' + name)
    model = model  # 建立模型
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    #score = model.score(x_test, y_test)
    #score = mean_squared_error(y_test,y_pred)
    score = r2_score(y_test,y_pred)
    print(score)
    score_.append(score)
    y_pre.append(y_pred)





# lr.fit(x_train,y_train)
# y_predict = lr.predict(x_test)
# print('线性回归  r2得分：',r2_score(y_test,y_predict))
# print('线性回归  均方差：',mean_squared_error(y_test,y_predict))
# print(y_test[:10])
# print(y_predict[:10])
#
plt.xlim([0,100])
plt.plot( range(len(y_test)), y_test, 'r', label='真实值')
plt.plot( range(len(y_pre[0])), y_pre[0], 'b--', label='线性回归' )
plt.plot( range(len(y_pre[2])), y_pre[2], 'g--', label='SVR' )
plt.title('sklearn: Linear Regression')
plt.ylabel('面积',size = 16)
plt.legend()
plt.show()