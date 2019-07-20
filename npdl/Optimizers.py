# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 22:23:25 2019
本程序为优化方法类，为模型提供参数优化的方法
@author: yanji
"""

import numpy as np
from . import Initial

class Optimizer():
    """
    构建所有优化器的父类，包括迭代速率，衰减系数，迭代速率锁定区间
    """
    def __init__(self,lr=0.001,clip=-1,decay=0.,lr_min=0.,lr_max=np.inf):
        self.lr = lr #初始迭代速率
        self.clip = clip #模型参数更新的最大边界
        self.decay = decay #迭代速率衰减系数
        self.lr_min = lr_min #迭代速率的最小值
        self.lr_max = lr_max #迭代速率的最大值
        self.iterations = 0 #模型的迭代次数，默认置0
        
    def update(self,params=None,grads=None):
        """
        Function:对模型权重截距等矩阵参数、迭代速率进行更新
        Parameters: params - 模型中的权重、截距等所有参数
                    grads - 模型参数对应的本次更新的微分
        Return: 无
        """
        self.iterations +=1 #模型迭代次数加1
        self.lr *= 1./(1+self.decay*self.iterations) #更新模型迭代速率
        self.lr = np.clip(self.lr,self.lr_min,self.lr_max) #锁定迭代速率在设定区间
        
class SGD(Optimizer):
    """
    构建随机梯度下降优化器类
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs) #调用父类初始化方法
    
    def update(self,params=None,grads=None):
        """
        Function:重构模型参数更新函数
        Parameters: params - 模型参数
                    grads - 模型参数对应的微分
        Return: 无
        """
        
        for p,g in zip(params,grads): #将参数与参数微分打包遍历
            if self.clip >0: #如果设定边界大于0，则将更新量锁定在边界区间内
                np.clip(g,-self.clip,self.clip) #锁定更新量
            p -= self.lr*g  #依据梯度负方向对参数进行更新
        super().update(params,grads) #调用父类的参数更新函数
            
def get(optimizer):
    """
    Function:依据表征优化方法的字符串，返回相应的优化器
    Parameters: optimizer - 表征优化方法的字符串
    Return: 相应的优化方法类
    """
    if optimizer in ['sgd','SGD']:
        return SGD() #如果选择的优化方法为随机梯度下降法，则返回SGD类
    else:
        raise ValueError('Unknown optimizer name:{}'.format(optimizer)) #如果为未知方法，引发未知优化器异常
                
            