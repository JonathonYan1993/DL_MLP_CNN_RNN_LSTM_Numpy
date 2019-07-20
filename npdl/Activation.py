# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:36:53 2019
本程序构建激活函数类
@author: yanji
"""

import numpy as np

class Activation():
    """
    构建激活函数父类
    """
    def __init__(self):
        self.last_forward=None #初始化本激活函数的输入为None,即上一层的前向传递输出
        
    def forward(self,input):
        """
        Function: 计算本激活函数的结果，即前向传递的值
        Parameters: input - 激活函数的输入
        Return: 父类无反馈，返回未实施错误
        """
        raise NotImplementedError #父类无反馈，返回未实施错误
    
    def derivative(self,input=None):
        """
        Function: 计算本激活函数对输入的微分
        Parameters: input - 激活函数输入
        Return: 父类无反馈，返回未实施错误
        """
        return NotImplementedError #父类无反馈，返回未实施错误

class ReLU(Activation):
    """
    构建整流线性激活函数类，整流线性单元以0为阈值，大于0通过，否则置0
    """
    def __init__(self):
        super().__init__() #调用父类Activation的初始化函数
    
    def forward(self,input):
        """
        Function: 重构前向传递函数，计算本激活函数的结果
        Parameters: input - 激活函数的输入
        Return: 激活函数的计算结果
        """
        self.last_forward = input
        return np.maximum(0.0,input)  #以0为阈值返回激活结果
    
    def derivative(self,input=None):
        """
        Function: 计算ReLU激活函数对输入的微分
        Parameters: input - 激活函数的输入
        Return: res - 整流线性激活函数对输入的微分
        """
        last_forward= input if input else self.last_forward #将激活层输入赋值给last_forward
        res= np.zeros(last_forward.shape,dtype='float32') #零值初始化微分矩阵
        res[last_forward>0]=1. #大于阈值0的输入参数的导数为1
        return res #返回激活函数对输入的微分
    
class Softmax(Activation):
    """
    构建Softmax激活函数类,本类作为最终Softmax输出层的激活函数，输出的是各个类别的概率
    """
    def __init__(self):
        super().__init__() #调用父类Activation的激活函数
    
    def forward(self,input):
        """
        Function:重构前向传递函数，计算本激活函数的结果
        Parameters: input - 激活函数的输入
        Return: act_output - 激活函数的输出结果
        """
        assert len(input.shape)==2 #断言激活层输入的维度为2，即数一维为数据序列，另一维为每一个数据
        self.last_forward = input #将输入数据存入激活函数输入
        input_Nomal= input - np.max(input,axis=1,keepdims=True) #归一化，将每个数据样本减去其类别数值的最大值，如此保证后续求取指数时维持在(0,1]之间
        exp_input = np.exp(input_Nomal) #对归一化的数据取指数
        act_output = exp_input/np.sum(exp_input,axis=1,keepdims=True) #计算softmax函数的输出，即求取每个类别的概率
        return act_output #返回激活函数的输出
    
    def derivative(self,input=None):
        """
        Function:计算本激活层对输入数据的微分
        Parameters: input - 激活函数的输入
        Return: 激活函数对输入的微分
        """
        last_forward = input if input else self.last_forward #有输入则用输入数据，无输入则用保存的输入数据
        #本激活函数直接输出最终类别概率，由于通过softmax交叉熵损失函数可以直接计算得到对本激活层输入的微分
        #则此处将传递的导数置1，参考博客：https://blog.csdn.net/hearthougan/article/details/82706834
        return np.ones(last_forward.shape, dtype='float32')  #返回本激活函数传递的导数
    
class Tanh(Activation):
    """
    构建tanh双曲正切激活函数类，用于RNN等隐藏层的激活
    """
    def __init__(self):
        super().__init__() #调用父类的初始化函数
        
    def forward(self,input):
        """
        Function:重构前向传递函数，计算本激活函数的结果
        Parameters: input - 激活函数的输入
        Return: 激活函数的输出结果
        """
        self.last_forward = np.tanh(input) #计算input的双曲正切结果，该结果直接作为激活函数的输出，同时保存在last_forward中
        return self.last_forward #返回激活函数的输出
    
    def derivative(self,input=None):
        """
        Function:计算本激活层对输入的微分
        Parameters: input -激活函数的输入
        Return: 激活函数对输入的微分
        """
        last_forward = self.forward(input) if input else self.last_forward #计算tanh(input)，该值用于微分计算
        #tanh(x)对x的导数为1-tanh(x)^2
        return 1-np.power(last_forward,2) #返回激活函数对输入的微分
    
class Sigmoid(Activation):
    """
    构建sigmoid激活函数类，用于LSTM隐藏层门结构的激活
    """
    def __init__(self):
        super().__init__() #调用父类的初始化函数
        
    def forward(self,input,*args,**kwargs):
        """
        Function:重构前向传递函数，计算本激活函数的结果
        Parameters: input - 激活函数的输入
                    *args - 多余的非关键字参数作为元组传递
                    **kwargs - 多余的关键字参数作为字典传递
        Return: 激活函数的输出结果
        """
        self.last_forward = 1.0/(1.0+np.exp(-input)) #计算sigmoid函数结果，保存到last_forward中
        return self.last_forward #返回激活函数的输出
    
    def derivative(self,input=None):
        """
        Function: 计算本激活层对输入的微分
        Parameters: input  - 激活函数的输入
        Return: 激活函数对输入的微分
        """
        last_forward = self.forward(input) if input else self.last_forward #如果有输入计算输入的前向传递结果，否则取保存的结果
        return np.multiply(last_forward,1-last_forward) # 计算并返回激活函数对输入的微分
        
def get(activation):
    """
    Function:调用激活函数的函数，依据字符串参数，返回对应的激活函数的类，以构建激活函数实例
    Parameters: activation -描述所采用激活函数的字符串
    Return:  激活函数类
    """
    if activation in ['relu','ReLU','RELU']:
        return ReLU() #如果所用激活函数为整流线性函数，返回整流线性激活函数
    if activation in ['softmax','Softmax']:
        return Softmax() #如果所用激活函数为softmax激活函数，返回softmax激活函数
    if activation in ['tan','tanh','Tanh']:
        return Tanh() #如果所用激活函数为双曲正切函数，返回Tanh激活函数类
    if activation in ['sigmoid','Sigmoid']:
        return Sigmoid() #如果所用激活函数为sigmoid函数，返回sigmoid激活函数类
    else:
        raise ValueError('Unknown activation name:{}.'.format(activation)) #如果为未知的激活函数类型，引发数值错误

        
