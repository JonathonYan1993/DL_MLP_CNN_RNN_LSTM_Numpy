# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 20:36:23 2019
本程序通过numpy构建神经网络模型，基于LSTM实现简单的语句词性分析
@author: yanji
"""

import os
import numpy as np
import npdl

def one_hot(dataMat,nb_classes=None):
    """
    Function: 将数据标签进行一维向量化，数据真实类别位置置1，其他位置置0
    Parameters: dataMat - 需要向量化的数据
                nb_classes - 数据类别种类
    Return: 一维向量化的数据
    """
    assert nb_classes is not None #断言nb_classes不为None
    if len(dataMat.shape) ==2:
        #如果为二维矩阵，第二维是序列数据，需要将每个数据进行一维向量化
        #零值初始化一维向量化数据
        one_hot_data= np.zeros((dataMat.shape[0],dataMat.shape[1],nb_classes),dtype='float32')
        for data_I in np.arange(dataMat.shape[0]):#遍历每个样本数据
            for time_I in np.arange(dataMat.shape[1]):#遍历每个序列数据
                labelNow = int(dataMat[data_I,time_I]) #该数据的类别标签
                one_hot_data[data_I,time_I,labelNow]=1 #将该位置置1
        return one_hot_data #返回一维向量化数据
    elif len(dataMat.shape)==1:
        #如果为一维矩阵，则将每个数据进行一维向量化
        #零值初始化一维向量化数据
        one_hot_data = np.zeros((dataMat.shape[0],nb_classes))
        for data_I in np.arange(dataMat.shape[0]): #遍历每个样本数据
            labelNow = int(dataMat[data_I]) #该数据的类别标签
            one_hot_data[data_I,labelNow]=1 #将该位置置1
        return one_hot_data #返回一维向量化数据
    else:
        raise NotImplementedError #引发未实施错误

if __name__ =='__main__':
    max_iter = 20 #最大迭代次数
    nb_batch = 30 #批量数据个数
    nb_seq = 20 #序列数据个数
    all_xs = []  #初始化样本数据
    all_ys = [] #初始化样本数据标签
    all_words = set() #初始化词汇集合
    all_labels = set() #初始化语句标签集合
    print('preparing data ...')
    #从文件中取出样本数据，词汇数据，标签数据
    with open(os.path.join(os.path.dirname(__file__),'data/trec/TREC_10.label')) as fin:
        for line in fin.readlines(): #遍历文件每一行
            words = line.strip().split() #去掉前后空格并进行分词
            y = words[0].split(':')[0] #取出该样本数据的标签
            xs = words[1:] #该样本数据的词汇序列数据
            all_xs.append(xs) #扩展样本数据列表
            all_ys.append(y) #扩展样本数据标签列表
            for word in words:
                all_words.add(word) #遍历词汇序列数据，扩展词汇集合
            all_labels.add(y) #扩展标签集合
    #给每个词汇加上索引
    word2idx = {w:i for i,w in enumerate(sorted(all_words))}
    #给每个标签加上索引
    label2idx = {label:i for i,label in enumerate(sorted(all_labels))}
    #将样本数据列表和标签列表变成索引列表
    all_idx_xs = []
    for sen in all_xs: #遍历每一条样本句子
        idx_x = [word2idx[word] for word in sen[:nb_seq]] #取出nb_seq个序列数据索引
        idx_x = [0]*(nb_seq-len(idx_x))+idx_x #不足nb_seq个数据的样本以0补充
        all_idx_xs.append(idx_x) #扩展样本数据索引列表
    all_idx_xs = np.array(all_idx_xs,dtype='float32') #转化为数组
    all_idx_ys = np.array([label2idx[label] for label in all_ys],dtype='float32') #标签索引列表
    X_size = len(word2idx) #词汇集合个数
    Y_size = len(label2idx) #标签集合个数
    X_data = one_hot(all_idx_xs,X_size) #将样本数据矩阵进行one_hot向量化
    Y_data = one_hot(all_idx_ys,Y_size) #将样本数据标签进行one_hot向量化
    print('building model ...')
    net = npdl.Model.Model() #初始化神经网络模型
    net.add(npdl.Layers.Embedding(nb_batch=nb_batch,nb_seq=nb_seq,n_out=200,input_size=X_size))#添加嵌入层，对数据降维
    net.add(npdl.Layers.LSTM(n_out=400,return_sequence=True)) #添加LSTM层
    net.add(npdl.Layers.LSTM(n_out=200,return_sequence=True)) #添加LSTM层
    net.add(npdl.Layers.MeanPooling((nb_seq,1))) #添加池化层
    net.add(npdl.Layers.Flatten()) #添加平坦层
    net.add(npdl.Layers.Softmax(n_out=Y_size)) #softmax输出层
    net.compile() #构建神经网络
    print('training model ...')
    net.fit(X_data,Y_data,batch_size=nb_batch,validation_split=0.1,max_iter=max_iter) #训练神经网络

    

            
        
        
        