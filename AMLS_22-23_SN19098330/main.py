import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,precision_score
import pandas as pd
import os
from PIL import Image

#Task A1
train_root="Dataset/dataset_AMLS_22-23/celeba"
test_root="Dataset/dataset_AMLS_22-23_test/celeba_test"
df = pd.read_csv(train_root+"/labels.csv")
Y_train=[]
for i in range(5000):
    li=df["\timg_name\tgender\tsmiling"][i].split('\t')
    Y_train.append(int(li[2]))
Y_train=np.array(Y_train)
Y_train=Y_train.reshape(-1,1)
X_train=[]
for j in range(5000):
    li2=df["\timg_name\tgender\tsmiling"][j].split('\t')
    path=train_root+"/img/"+li2[1]
    im=Image.open(path).convert("RGB")
    im = im.resize((100, 100))
    ii=np.asarray(im)
    ii=ii.flatten()
    X_train.append(ii)
X_train=np.array(X_train)
df_t=pd.read_csv(test_root+"/labels.csv")
Y_test=[]
for i in range(1000):
    li=df_t["\timg_name\tgender\tsmiling"][i].split('\t')
    Y_test.append(int(li[2]))
Y_test=np.array(Y_test)
Y_test=Y_test.reshape(-1,1)
X_test=[]
for j in range(1000):
    li2=df_t["\timg_name\tgender\tsmiling"][j].split('\t')
    path=test_root+"/img/"+li2[1]
    im=Image.open(path).convert("RGB")
    im = im.resize((100, 100))
    ii=np.asarray(im)
    ii=ii.flatten()
    X_test.append(ii)
X_test=np.array(X_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=1000)#PCA dimension reduction, change n_component to affect the accuracy
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)
model=SVC()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
print(accuracy_score(Y_test,Y_pred))
#Task A2
Y_train=[]
for i in range(5000):
    li=df["\timg_name\tgender\tsmiling"][i].split('\t')
    Y_train.append(int(li[3]))
Y_train=np.array(Y_train)
Y_train=Y_train.reshape(-1,1)
X_train=[]
for j in range(5000):
    li2=df["\timg_name\tgender\tsmiling"][j].split('\t')
    path=train_root+"/img/"+li2[1]
    im=Image.open(path).convert("RGB")
    im = im.resize((100, 100))
    ii=np.asarray(im)
    ii=ii.flatten()
    X_train.append(ii)
X_train=np.array(X_train)
df_t=pd.read_csv(test_root+"/labels.csv")
Y_test=[]
for i in range(1000):
    li=df_t["\timg_name\tgender\tsmiling"][i].split('\t')
    Y_test.append(int(li[3]))
Y_test=np.array(Y_test)
Y_test=Y_test.reshape(-1,1)
X_test=[]
for j in range(1000):
    li2=df_t["\timg_name\tgender\tsmiling"][j].split('\t')
    path=test_root+"/img/"+li2[1]
    im=Image.open(path).convert("RGB")
    im = im.resize((100, 100))
    ii=np.asarray(im)
    ii=ii.flatten()
    X_test.append(ii)
X_test=np.array(X_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=1000)#PCA dimension reduction, change n_component to affect the accuracy
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)
model=SVC()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
print(accuracy_score(Y_test,Y_pred))
#Task B1
train_root="Dataset/dataset_AMLS_22-23/cartoon_set"
test_root="Dataset/dataset_AMLS_22-23_test/cartoon_set_test"
df = pd.read_csv(train_root+"/labels.csv")
Y_train=[]
for i in range(10000):
    li=df["\teye_color\tface_shape\tfile_name"][i].split('\t')
    Y_train.append(int(li[2]))
Y_train=np.array(Y_train)
Y_train=Y_train.reshape(-1,1)
X_train=[]
for j in range(10000):
    li2=df["\teye_color\tface_shape\tfile_name"][j].split('\t')
    path=train_root+"/img/"+li2[3]
    im=Image.open(path).convert("RGB")
    im = im.resize((200, 200))
    ii=np.asarray(im)
    ii=ii.flatten()
    X_train.append(ii)
X_train=np.array(X_train)
df_t=pd.read_csv(test_root+"/labels.csv")
Y_test=[]
for i in range(2500):
    li=df_t["\teye_color\tface_shape\tfile_name"][i].split('\t')
    Y_test.append(int(li[2]))
Y_test=np.array(Y_test)
Y_test=Y_test.reshape(-1,1)
X_test=[]
for j in range(2500):
    li2=df_t["\teye_color\tface_shape\tfile_name"][j].split('\t')
    path=test_root+"/img/"+li2[3]
    im=Image.open(path).convert("RGB")
    im = im.resize((100, 100))
    ii=np.asarray(im)
    ii=ii.flatten()
    X_test.append(ii)
X_test=np.array(X_test)
from sklearn.decomposition import PCA
pca = PCA(n_components=500)#PCA dimension reduction, change n_component to affect the accuracy
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)
model=SVC()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
print(accuracy_score(Y_test,Y_pred))
#Task B2
df = pd.read_csv(train_root+"/labels.csv")
Y_train=[]

for i in range(10000):
    li=df["\teye_color\tface_shape\tfile_name"][i].split('\t')
    Y_train.append(int(li[1]))
Y_train=np.array(Y_train)
Y_train=Y_train.reshape(-1,1)
X_train=[]

for j in range(10000):
    li2=df["\teye_color\tface_shape\tfile_name"][j].split('\t')
    path=train_root+"/img/"+li2[3]
    im=Image.open(path).convert("RGB")
    im = im.resize((200, 200))
    ii=np.asarray(im)
    ii=ii.flatten()
    X_train.append(ii)
X_train=np.array(X_train)
df_t=pd.read_csv(test_root+"/labels.csv")
Y_test=[]

for i in range(2500):
    li=df_t["\teye_color\tface_shape\tfile_name"][i].split('\t')
    Y_test.append(int(li[1]))
Y_test=np.array(Y_test)
Y_test=Y_test.reshape(-1,1)
X_test=[]

for j in range(2500):
    li2=df_t["\teye_color\tface_shape\tfile_name"][j].split('\t')
    path=test_root+"/img/"+li2[3]
    im=Image.open(path).convert("RGB")
    im = im.resize((100, 100))
    ii=np.asarray(im)
    ii=ii.flatten()
    X_test.append(ii)
X_test=np.array(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=500) #PCA dimension reduction, change n_component to affect the accuracy
X_train=pca.fit_transform(X_train)
X_test=pca.fit_transform(X_test)
model=SVC()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
print(accuracy_score(Y_test,Y_pred))