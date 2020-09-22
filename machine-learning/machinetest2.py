import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import torch
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
def number():
    number1 = []
    for i in range(50):
        b = np.random.uniform(1, 50)
        b = round(b,2)

        number1.append(b)
    number1_x = np.arange(1,51).reshape(50,1)
    number1_y = np.array(number1).reshape(50,1)
    num1 = np.concatenate((number1_x, number1_y), axis=1)

    number2 = []
    for i in range(50):
        b = np.random.uniform(10, 30)
        b = round(b,2)

        number2.append(b)
    number2_x = np.arange(70,120).reshape(50,1)
    number2_y = np.array(number2).reshape(50,1)
    num2 = np.concatenate((number2_x, number2_y), axis=1)
    num3 = np.concatenate((num1, num2), axis=0)

    file = open(r'text1.txt', mode='w')
    for i in range(len(num3)):
        s = str(num3[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        s = s.strip() + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        file.write(s)
#number()

def delspace(file):
    txt = open(file, 'r')
    newtxt = open('new.txt', 'w')
    k = 0
    x = []
    lines = txt.readlines()
    for line in lines:
        if k<50:
            k+=1
            a = [str(i).strip('\n') for i in line]
            b = ''.join(a)
            x.append(b)
            newtxt.write(b+'  '+'0'+'\n')

        else:
            a = [str(i).strip('\n') for i in line]
            b = ''.join(a)  #拼接字符串
            x.append(b)
            newtxt.write(b + '  '+'1' + '\n')
    x = np.stack(x) #转成numpy
    print(type(x))
    txt.close()
    newtxt.close()
#delspace('text1.txt')
# y= torch.tensor([1,2,3,4,5])
# one_hot = torch.zeros(y.size(0),10).scatter_(1,y.reshape(-1,1),1)
# print(one_hot)
# a = np.loadtxt('new.txt')
# x1 = a[:50,0]
# y1 = a[:50,1]
# x2 = a[51:,0]
# y2 = a[51:,1]
# print(x2.shape)
# plt.figure()
# plt.scatter(x1,y1,c='r')
# plt.scatter(x2,y2)
# plt.show()
# data = np.random.rand(100,3)
# estime = KMeans(n_clusters=3)   #构建模型
# estime.fit_predict(data)    #聚类
# center = estime.cluster_centers_    #获取聚类中心点
# label_pred = estime.labels_     #获取类别的标签
#
# # fig = plt.figure()
# # ax = Axes3D(fig)
# # ax.scatter(data[:,0],data[:,1],data[:,2],c=label_pred,marker='*',s=40)
# # ax.scatter(center[:,0],center[:,1],center[:,2],marker='<',s=40)
# # plt.show()
# fig,ax = plt.subplots()
# ax.scatter(data[:,0],data[:,1],data[:,2],c=label_pred,marker='*')
# plt.show()

x,y=make_blobs(n_samples=300,random_state=1500)
estime = KMeans(n_clusters=6)
y_pred = estime.fit_predict(x)


#scatter绘制散点图，c指的是颜色，根据不同的类给不同的颜色
plt.subplot(121)
plt.scatter(x[:,0],x[:,1],c=y,marker='*',)  #套公式就行了
plt.title("kMeans01")
plt.subplot(122)
plt.scatter(x[:,0],x[:,1],c=y_pred,marker='*',)
plt.title("kMeans02")
plt.show()
