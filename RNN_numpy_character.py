# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 21:48:03 2019
本程序通过numpy构建神经网络模型，基于RNN实现简单的文本分类
@author: yanji
"""
import os
import numpy as np
import npdl

if __name__=='__main__':
    print('Preparing data ...')
    txtdata_path = os.path.join(os.path.dirname(__file__),'data/lm/tiny_shakespeare.txt') #文本数据路径
    raw_text = open(txtdata_path,'r').read() #读取文件内容
    chars = list(set(raw_text)) #列出文本不同字符
    data_size, vocab_size = len(raw_text), len(chars) #数据个数，不重复数据个数(类别种类数)
    print('data has %d characters, %d unique.'%(data_size,vocab_size)) #打印数据个数
    char_to_index = {ch:i for i,ch in enumerate(chars)} #数据编码
    index_to_char = {i:ch for i,ch in enumerate(chars)} #数据解码
    time_steps, batch_size =30, 40 #时间序列长度，单批数据大小
    length = batch_size*20 #用于训练的数据大小
    text_pointers = np.random.randint(data_size-time_steps-1,size=length) #随机抽取length个数据的初始索引位置
    batch_in = np.zeros((length,time_steps,vocab_size)) #初始化输入数据矩阵,length个数据个数，每个数据为time_steps个序列数据，每个序列数据有vocab_size个索引
    batch_out = np.zeros((length,vocab_size),dtype=np.uint8) #初始化输入数据标签
    for i in np.arange(length):
        b_ = [char_to_index[c] for c in raw_text[text_pointers[i]:text_pointers[i]+time_steps+1]] #取出第i个数据的序列数据
        batch_in[i,range(time_steps),b_[:-1]]=1 #将序列数据中，每个时刻数据对应的标签编码位置置1
        batch_out[i,b_[-1]]=1 #将序列第time_steps+1数据的真实标签位置置1，作为该序列数据的最终类别标签，即由前面time_steps个数据标签可以推断出time_steps+1时刻的数据标签
    
    print('Building model ...')
    net=npdl.Model.Model() #初始化神经网络模型
    net.add(npdl.Layers.SimpleRNN(n_out=200,n_in=vocab_size,nb_batch=batch_size,nb_seq=time_steps,return_sequence=True)) #添加RNN网络层
    net.add(npdl.Layers.SimpleRNN(n_out=200,return_sequence=True)) #添加RNN网络层
    net.add(npdl.Layers.MeanPooling(pool_size=(time_steps,1))) #添加池化层
    net.add(npdl.Layers.Flatten()) #添加平坦层
    net.add(npdl.Layers.Softmax(n_out=vocab_size)) #添加Softmax输出层
    net.compile() #将个网络层进行组合
    
    max_iter=50 #最大迭代次数
    print('Training model ...')
    net.fit(batch_in,batch_out,max_iter=max_iter,batch_size=batch_size) #训练模型
        
        
    
    