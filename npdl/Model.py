# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 18:38:02 2019
本程序为构建模型类，通过调用本程序中Model类，构建神经网络模型
@author: yanji
"""

import numpy as np 
from .Layers import Layer
from . import Optimizers
from . import Cost

class Model():
    """
    定义神经网络模型类，包括网络层添加、网络层组合、训练、预测、精度求取函数
    """
    def __init__(self,layers=None):
        self.layers = [] if layers is None else layers #初始化模型的神经网络层序列，None则为空
        self.loss = None #初始化模型的损失函数
        self.optimizer = None #初始化模型的优化方法
        
    def add(self,layer):
        """
        Function: 网络层添加函数，给模型的神经网络序列添加网络层
        Parameters: layer - 网络层类，父类同为Layer
        Return: 无
        """
        assert isinstance(layer,Layer),"Must be 'Layer' instance" #断言layer是不是网络层Layer类
        self.layers.append(layer) #模型的神经网络序列扩展该网络层
        
    def compile(self,loss='SoftmaxCategoricalCrossEntropy',optimizer='SGD'):
        """
        Function:将各层神经网络进行连接组合
        Parameters: loss - 神经网络模型选用的损失函数，默认为softmax交叉熵损失函数
                    optimizer - 神经网络模型选用的优化方法，默认为随机梯度下降法
        Return: 无
        """
        next_layer = None #初始化前面一层为None
        for layer in self.layers: #遍历神经网络每一层
            layer.connect_to(next_layer) #将神经网络每层与前面一层连接
            next_layer = layer #更新表征前面一层的变量
        self.loss = Cost.get(loss) #得到模型的损失函数类
        self.optimizer = Optimizers.get(optimizer) #得到模型的优化方法类
    
    def predict(self,X):
        """
        Function:对输入的数据X进行预测
        Parameters: X - 要预测的数据，单个数据或者数据矩阵
        Return: y_pred - 预测结果
        """
        x_next = X #初始化要传递给后面一层神经网络的数据矩阵
        for layer in self.layers:
            x_next=layer.forward(x_next) #逐层前向传递计算传递给后面一层神经网络的数据
        y_pred = x_next #最后一层的输出即为数据X的预测结果
        return y_pred #返回预测结果
    
    def accuracy(self,outputs,targets):
        """
        Function: 计算预测结果的准确率
        Parameters: outputs -预测类别结果
                    targets - 真实类别标签
        Return:  预测准确率
        """
        y_predicts = np.argmax(outputs,axis=1) #得到每个数据预测结果最大可能性的类别索引值
        y_targets = np.argmax(targets,axis=1) #得到每个数据真实结果的类别索引值
        acc = y_predicts==y_targets #预测结果与真实结果比较，存为bool序列传给acc
        return np.mean(acc)  #取acc的平均值即得到预测的准确率
    
    def fit(self,X,Y,max_iter=100,batch_size=64,
            validation_split=0.,validation_data=None):
        """
        Function:训练神经网络模型
        Parameters: X - 数据集属性矩阵
                    Y - 数据集类别标签
                    max_iter - 最大迭代次数，即训练轮数
                    batch_size - 随机梯度下降法的单批数据批次大小
                    validation_split - 验证集划分比例
                    validation_data - 验证集数据
        Return: 无
        """
        train_X = X.astype('float32') #将数据矩阵转换成单精度浮点数
        train_Y = Y.astype('float32') #将数据标签转换成单精度浮点数
        if 1.>validation_split>0.: #如果验证集划分比例在0到1之间，可以进行划分
            split = int(train_Y.shape[0]*validation_split) #计算划分到验证集的数据个数
            valid_X, valid_Y = train_X[-split:], train_Y[-split:]  #划出验证集
            train_X, train_Y = train_X[:-split], train_Y[:-split] #剩下的作为训练集
        elif validation_data is not None: #如果验证集不为None，取参数传递的验证集
            valid_X, valid_Y = validation_data #取出验证集
        else:
            valid_X, valid_Y = None, None #否则没有验证集，置为None
        
        iter_idx =0 #初始化迭代次数为0
        while iter_idx < max_iter: #如果迭代次数小于最大迭代次数，继续迭代
            iter_idx +=1 #迭代次数更新
            seed=np.random.randint(111,1111111) #生成一个随机数种子
            np.random.seed(seed) #使用这个随机数种子
            np.random.shuffle(train_X) #随机打乱训练集数据
            np.random.seed(seed) #再次使用这个随机数种子
            np.random.shuffle(train_Y) #随机打乱训练集标签
            train_loss, train_predicts, train_targets = [], [], [] #初始化本轮损失函数值，预测结果，真实结果
            for num_batch in np.arange(train_Y.shape[0]//batch_size): #计算本轮要训练的批次，遍历每一批
                batch_begin = num_batch*batch_size #本批次数据索引初始点
                batch_end = batch_begin + batch_size #本批次数据索引终止点
                x_batch = train_X[batch_begin:batch_end] #取出本批次数据
                y_batch = train_Y[batch_begin:batch_end] #取出本批次数据标签
                y_pred = self.predict(x_batch) #对该批次数据进行结果预测
                next_grad = self.loss.backward(y_pred,y_batch) #计算损失函数对预测结果的梯度，作为梯度方向传播的开始
                for layer in self.layers[::-1]: #反向遍历每一层，进行梯度方向传播
                    next_grad = layer.backward(next_grad) #反向遍历每一层，计算向前面一层传递的梯度
                params=[] #初始化模型参数列表
                grads =[] #初始化模型参数梯度列表
                for layer in self.layers: #遍历每一层
                    #layer.params[0] -=self.optimizer.lr*layer.grads[0] #更新每一层的权重和截距参数
                    #layer.params[1] -=self.optimizer.lr*layer.grads[1] #更新每一层的截距参数
                    params.extend(layer.params) #扩展模型参数矩阵列表
                    grads.extend(layer.grads) #扩展模型参数梯度矩阵列表
                self.optimizer.update(params,grads) #更新模型参数，同时更新迭代速率
                train_loss.append(self.loss.forward(y_pred,y_batch)) #扩展本轮损失函数值
                train_predicts.extend(y_pred) #扩展本轮预测结果值
                train_targets.extend(y_batch) #扩展本轮真实结果值
            #本轮训练完需要输出的训练结果
            runout ="iter %d, train- [loss %.4f, acc %.4f];"%(
                    iter_idx, float(np.mean(train_loss)),float(self.accuracy(train_predicts,train_targets)))
            if valid_X is not None and valid_Y is not None: #如果存在验证集，需要对验证集进行预测
                valid_loss, valid_predicts, valid_targets =[],[],[]
                for num_batch in np.arange(valid_X.shape[0]//batch_size): #计算验证集要训练的批次，遍历每一批
                    batch_begin = num_batch*batch_size #本批次数据索引开始点
                    batch_end = batch_begin + batch_size #本批次数据索引终止点
                    x_batch = valid_X[batch_begin:batch_end] #取出本批次数据
                    y_batch = valid_Y[batch_begin:batch_end] #取出本批次数据标签
                    y_pred = self.predict(x_batch) #预测本批次数据
                    valid_loss.append(self.loss.forward(y_pred,y_batch)) #扩展验证集损失函数值
                    valid_predicts.extend(y_pred) #扩展验证集预测结果值
                    valid_targets.extend(y_batch) #扩展验证集真实结果值
                #添加验证集预测结果输出
                runout += "\nvalid-[loss %.4f, acc %.4f];"%(
                        float(np.mean(valid_loss)),float(self.accuracy(valid_predicts,valid_targets)))
            print(runout)  #将本轮训练结果输出
            
            
            
            
        
        
