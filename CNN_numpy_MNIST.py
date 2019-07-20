# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 21:33:39 2019
#本程序通过numpy实现简单的CNN神经网络，包括卷积、探测、池化以及反向传播算法等
#本程序通过CNN实现minist数据集的分类
@author: yanji
"""

import os 
import numpy as np
from sklearn.datasets import fetch_mldata
import npdl 

def one_hot(labels,nb_classes=None):
    """
    Function: 将数据标签进行一维向量化，数据真实类别置1，其他位置置0
    Parameters: labels - 需要向量化的数据标签
                nb_classes - 数据类别种类
    Return: 一维向量化的数据标签
    """
    classes = np.unique(labels) #取出所有不同的类别标签
    if nb_classes is None:
        nb_classes =classes.size #种类数
    one_hot_labels = np.zeros((labels.shape[0],nb_classes)) #初始化向量化标签矩阵
    for i,c in enumerate(classes):
        one_hot_labels[labels==c,i]=1 #将类别标签为c的数据的索引位置i置1
    return one_hot_labels #返回向量化数据标签
        

if __name__=='__main__':
    max_iter = 10 #最大迭代次数
    seed=100 #随机数种子
    nb_data =1000 #用于训练的数据个数
    
    print("loading mnist dataset ...")
    #取出mnist数据集
    mnist = fetch_mldata('MNIST original',data_home =os.path.join(os.path.dirname(__file__),'./data'))
    X_train = mnist.data.reshape((-1,1,28,28))/255.0 #取出数据，并进行归一化
    np.random.seed(seed) #设定随机数种子
    X_train = np.random.permutation(X_train)[:nb_data] #随机取nb_data个数据
    y_train = mnist.target #取出数据标签
    np.random.seed(seed) #使用同一随机数种子
    y_train = np.random.permutation(y_train)[:nb_data] #取出nb_data个数据对应标签
    n_classes = np.unique(y_train).size #总的分类类别数目
    one_hot_y_train = one_hot(y_train) #将类别标签进行一维向量化
    
    print('building model ...')
    net = npdl.Model.Model() #初始化神经网络模型
    net.add(npdl.Layers.Convolution(1,(3,3),input_shape=(100,1,28,28))) #添加卷积层为第一层
    net.add(npdl.Layers.MeanPooling((2,2))) #添加平均池化层
    net.add(npdl.Layers.Convolution(2,(4,4))) #添加卷积层
    net.add(npdl.Layers.MeanPooling((2,2))) #添加平均池化层
    net.add(npdl.Layers.Flatten()) #添加平坦层
    net.add(npdl.Layers.Softmax(n_out=n_classes)) #添加softmax输出层
    net.compile() #将模型各网络层进行组合
    
    print('training model ...')
    net.fit(X_train,one_hot_y_train,max_iter=max_iter,batch_size=100,validation_split=0.1) #训练模型
    
    

