# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 23:41:48 2019
本程序为代价函数类
@author: yanji
"""

import numpy as np

class Objective():
    """
    定义代价函数的基类，包括前向传递函数，梯度方向传播函数
    """
    def forward(self,outputs,targets):
        """
        Function:前向传递函数，由预测结果与真实结果进行比较，计算代价函数值
        Parameters: outputs - 预测结果值
                    targets - 真实结果值
        Return: 代价函数值
        """
        raise NotImplementedError #父类无反馈，引发未实施错误
        
    def backward(self,outputs,targets):
        """
        Function: 梯度反向传播函数，计算代价函数对预测输出结果的梯度
        Parameters: outputs - 预测结果值
                    targets - 真实结果值
        Return: 代价函数对预测输出结果的梯度
        """
        raise NotImplementedError #父类无反馈，引发未实施错误
        
class SoftmaxCategoricalCrossEntropy(Objective):
    """
    定义softmax函数交叉熵损失函数类
    参考博客https://blog.csdn.net/hearthougan/article/details/82706834
    """
    def __init__(self,epsilon=1e-11):
        self.epsilon =epsilon #各类类别概率离边界0/1的最小距离
    
    def forward(self,outputs,targets):
        """
        Function:重构前向传递函数，由预测结果与真实结果进行比较，计算损失函数值
        Parameters: outputs - 预测结果值
                    targets - 真实结果值
        Return: 代价函数值
        """
        clip_output = np.clip(outputs,self.epsilon,1-self.epsilon) #将各类别概率锁定在此区间
        return np.mean(-np.sum(targets*np.log(clip_output),axis=1)) #计算损失函数值
    
    def backward(self,outputs,targets):
        """
        Function:重构梯度反向传播函数，计算代价函数对预测输出结果的梯度
        Parameters: outputs - 预测结果值
                    targets - 真实结果值
        Return: 代价函数对预测输出结果的梯度
        """
        clip_output = np.clip(outputs,self.epsilon,1-self.epsilon) #将各类别概率锁定在此区间
        #此处代价函数返回的实为代价函数对softmax层的全连接层的输出数据的梯度，即为传递到softmax函数输入的梯度
        return clip_output - targets  #返回代价函数传递到激活层输出的梯度

def get(objective):
    """
    Function:依据表征代价函数的字符串信息，返回代价函数类
    Parameters: objective - 表征代价函数的字符串
    Return: 代价函数类
    """
    if objective in ['softmax_categorical_cross_entropy', 'SoftmaxCategoricalCrossEntropy']:
        return SoftmaxCategoricalCrossEntropy() #如果为softmax交叉熵函数，则返回softmax损失函数类
    else:
        raise ValueError('Unknown Cost name:{}'.format(objective)) #如果为未知的代价函数，则返回未知异常
