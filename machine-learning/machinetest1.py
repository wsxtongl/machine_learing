from sklearn import datasets,neighbors,preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
import torch
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
import json
# image = sitk.ReadImage(r'D:\lumbar\lumbar_testA50\study201\image1.dcm')
# image_array = np.squeeze(sitk.GetArrayFromImage(image))
# print(image)
# plt.imshow(image_array,'gray')
# plt.show()
# a = open(r'./lumbar_train51_annotation.json')
# b = json.load(a)
# i = 0
# for c in b:
#
#     name = c['data']
#     print(name)

# iris = datasets.load_iris()
# x,y = iris.data,iris.target
# #划分
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
#
# #标准化处理
# scaler = preprocessing.StandardScaler()
#
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.fit_transform(x_test)

# #创建模型
# knn = neighbors.KNeighborsClassifier(n_neighbors=5)
# #模型拟合
# knn.fit(x_train,y_train)
# #交叉验证
# score = cross_val_score(knn,x_train,y_train,cv=5,scoring="accuracy")
#
# y_predict = knn.predict(x_test)
# print(accuracy_score(y_test,y_predict))
# iris = datasets.load_iris()
# x,y = iris.data,iris.target
# #标准化处理
# # scaler = preprocessing.StandardScaler()
# #
# # x_train = scaler.fit_transform(x)
# # x_test = scaler.fit_transform(y)
# k_x = range(1,31)
# k_score = []
# for i in k_x:
#     knn = neighbors.KNeighborsClassifier(n_neighbors=i)
#     scores = cross_val_score(knn,x,y,cv=5,scoring="accuracy")
#     k_score.append(scores.mean())
# plt.figure()
# plt.plot(k_x,k_score)
# plt.show()


#one-hot编码
# def Tenhot():
#     str_number = input("请输入:")
#     np_number= np.zeros(10,np.int8)
#     np_number[int(str_number)]=1
#     str_num = ''.join(str(i) for i in np_number)
#     print(str_num)
#     return str_num
# str_num= Tenhot()
# def number(str):
#     b= list(str)
#     k = -1
#     for i in b:
#         k+=1
#         if 1 == int(i):
#             print(k)
#             return k
# def one_host():
#
#     number = torch.tensor([1,3])
#     print(number.reshape(-1,1))
#     a= torch.zeros(number.size(0),10).scatter_(1,number.reshape(-1,1),1)
#     print(a)
# one_host()
