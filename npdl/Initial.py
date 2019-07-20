# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 21:13:29 2019
本程序为初始化类库，通过调用本库中的初始化类，对数据矩阵进行相应的初始化操作
@author: yanji
"""

import numpy as np

class Initializer():
    """
    构建初始化类的一个基类，用于构建初始化类的结构
    """
    def __call__(self,size):
        """
        Function:将类的实例初始化为一个可调用对象，依据输入参数size返回相应的初始化结果
        Parameters: size - 要初始化的参数矩阵的尺寸形状
        Return: 初始化后的参数矩阵
        """
        return self.call(size)  #进一步调用类的内置函数，返回初始化后的参数矩阵
    
    def call(self,size):
        """
        Function: 依据输入参数size返回相应的初始化结果
        Parameters: size -要初始化的参数矩阵的尺寸形状
        Return: 初始化后的参数矩阵
        """
        raise NotImplementedError #作为基类，不反馈，引发未实施异常
    
class Zero(Initializer):
    """
    定义零值初始化类，用于零值初始化参数矩阵，该类继承初始化基类Initializer
    """
    def call(self,size):
        """
        Function:重构call函数，返回零值初始化矩阵
        Parameters: size - 要初始化的参数矩阵的尺寸形状
        Return: 零值初始化后的参数矩阵
        """
        return np.array(np.zeros(size),dtype='float32') #返回size大小的零值初始化矩阵

class Uniform(Initializer):
    """
    定义均匀分布初始化类
    """
    def __init__(self,scale=0.05):
        """
        定义初始化函数
        Parameter: scale - 随机初始化的边界
        """
        self.scale=scale #随机初始化的边界
        
    def call(self,size):
        """
        Function:重构call函数，返回均匀分布随机初始化矩阵
        Parameters: size - 要初始化的参数矩阵的尺寸形状
        Return: 均匀分布的随机初始化的参数矩阵
        """
        return np.array(np.random.uniform(-self.scale,self.scale,size=size),dtype='float32') #返回随机初始化后的参数矩阵
    
        
 
class He_Uniform(Initializer):
    """
    定义He均匀分布初始化类，常在激活函数为ReLU的情况下使用
    """
    def call(self,size):
        """
        Function:重构call函数，返回均匀分布随机初始化矩阵
        Parameters: size - 要初始化的参数矩阵的尺寸形状
        Return: 均匀分布随机初始化后的参数矩阵
        """
        fan_in = size[1]*np.prod(size[2:]) #输入的参数个数
        scale=np.sqrt(6./fan_in) #计算随机取值的边界
        return np.array(np.random.uniform(-scale,scale,size=size),dtype='float32') #返回随机初始化的参数矩阵
    
class Glorot_Uniform(Initializer):
    """
    定义Glorot均匀分布初始化类，适合激活函数为tanh的情况，不适合激活函数为sigmoid和relu的情况
    """
    def call(self,size):
        """
        Function:重构call函数，返回均匀分布的随机初始化矩阵
        Parameters: size - 要初始化的参数矩阵的尺寸形状
        Return: 均匀分布随机初始化后的参数矩阵
        """
        fan_in = size[1]*np.prod(size[2:]) #输入的参数个数
        fan_out = size[0]*np.prod(size[2:]) #输出的参数个数
        scale = np.sqrt(6./(fan_in+fan_out)) #计算随机取值的边界
        return np.array(np.random.uniform(-scale,scale,size=size),dtype='float32') #返回随机初始化的参数矩阵

class Orthogonal(Initializer):
    """
    定义正交初始化类，避免RNN在训练初始发生梯度爆炸或消失现象
    """
    def __init__(self,gain=1.0):
        if gain =='relu':
            gain=np.sqrt(2) #默认增益为1，采用relu激活函数时为根号2
        self.gain = gain #正交初始化类的增益
    
    def call(self,size):
        """
        Function:重构call函数，返回正交随机初始化矩阵
        Parameters: size -要初始化的参数矩阵的尺寸形状
        Return: 正交随机初始化的参数矩阵
        """
        flat_shape = (size[0],np.prod(size[1:])) #先将参数尺寸形状平坦为2维
        a = np.random.normal(loc=0.,scale=1.,size=flat_shape) #生成高斯分布的随机数矩阵
        u,_,v = np.linalg.svd(a, full_matrices=False) #对随机数矩阵进行奇异值分解
        q = u if u.shape == flat_shape else v #取u,v中形状为flat_shape的参数矩阵
        q=q.reshape(size) #将参数矩阵还原成size形状
        q=self.gain*q #参数矩阵放大增益
        return np.array(q,dtype='float32') #返回随机初始化后的参数矩阵
    
def get(initialization):
    """
    Function:定义返回初始化类的函数，依据描述初始化方法的字符串返回相应的初始化类
    Parameters: initialization - 描述初始化方法的字符串
    Return: 返回初始化类
    """
    if initialization in ['zero','Zero']:
        return Zero()  #如果为零值初始化，返回零值初始化类
    elif initialization in ['uniform','Uniform']:
        return Uniform()#如果为uniform初始化，返回Uniform初始化类
    elif initialization in ['HeUniform','he_uniform']:
        return He_Uniform() #如果为He_Uniform初始化，返回He_Uniform初始化类
    elif initialization in ['glorot_uniform','GlorotUniform']:
        return Glorot_Uniform()# 如果为Glorot_Uniform初始化，返回Glorot_Uniform初始化类
    elif initialization in ['Orthogonal','orthogonal']:
        return Orthogonal() #如果为正交初始化，返回Orthogonal初始化类
    else:
        raise ValueError('Unknown initialization name:{}.'.format(initialization) ) #若没有找到初始化类型，引发数值错误

    
                
        