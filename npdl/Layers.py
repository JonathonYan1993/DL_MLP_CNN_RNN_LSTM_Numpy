# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 20:12:32 2019
本程序旨在建立Layer层类库，
包括作为父类的base层类，
需要用到的convolution卷积层类，
meanpooling池化层类，
维度降到1维输出的Flattern层类，
softmax函数
输出的Softmax层类
参考资料：《深度学习》
https://github.com/oujago/NumpyDL/tree/master/npdl/layers
@author: yanji
"""

import numpy as np
from . import Initial
from . import Activation

class Layer():
    """
    本类作为神经网络各层的父类，构建前向传递，反向传播，层间连接函数，层的参数与梯度属性
    """
    def forward(self,input,*args,**kwargs):
        """
        Function:前向传递函数
        Parameters: input - 输入
                    *args - 多余的非关键字参数存为元组传入
                    **kwargs - 多余的关键字参数存为字典传入
        Return: 父类无反馈，引发NotImplementedError错误
        """
        raise NotImplementedError  #引发未实施错误
        
    def backward(self,pre_grad,*args,**kwargs):
        """
        Function: 梯度反向传播
        Parameters: pre_grad - 上面一层的梯度
                    *args - 多余的非关键字参数存为元组传入
                    **kwargs - 多余的关键字参数存为字典传入
        Return: 父类无反馈，引发NotImplementedError错误
        """
        raise NotImplementedError #引发未实施错误
        
    def connect_to(self,prev_layer):
        """
        Function: 将本层网络与前面一层网络连接起来
        Parameters: prev_layer - 前面一层网络，同为Layer类
        Return : 父类无反馈，引发NotImplementedError错误
        """
        raise NotImplementedError #引发未实施错误
        
    @property  #将参数作为属性调用
    def params(self):
        """
        Function: 返回本层参数
        Parameters: 无
        Returns: 返回参数列表
        """
        return [] #如果该层没有参数，返回空列表
    
    @property #将梯度作为属性调用
    def grads(self):
        """
        Function: 返回本层参数梯度
        Parameters: 无
        Returns: 返回参数梯度列表
        """
        return [] #如果该层没有参数，返回空列表
    
    def __str__(self):
        """
        Function: 魔法方法，以字符串形式返回该函数return的结果
        Parameter: 无
        Returns: __class__.__name__ - 类名
        """
        return self.__class__.__name__  #返回该类的类名
    
    
class Convolution(Layer):
    """
    本类为卷积层类
    """
    def __init__(self,nb_filter,filter_size,input_shape=None,stride=1,
                 init='glorot_uniform',activation='relu'):
        self.nb_filter = nb_filter #卷积滤波器的个数
        self.filter_size = filter_size #卷积滤波器的大小
        self.input_shape = input_shape #上一层输出数据的尺寸形状,即层的输入形状
        self.stride =stride #卷积滤波器的步幅，默认为1
        self.W, self.dW = None, None #初始化卷积层的权重、权重微分为None
        self.b, self.db = None, None #初始化卷积层的偏置、偏置微分为None
        self.out_shape = None #初始化卷积层输出尺寸为None
        self.last_output = None #初始化卷积层上一次的输出
        self.last_input = None #初始化卷积层上一次的输入
        self.init = Initial.get(init) #得到卷积层的参数初始化类
        self.activation = Activation.get(activation) #得到卷积层的激活函数
        
    def connect_to(self,pre_layer=None):
        """
        Function:重构网络连接函数，将本层与上一层连接起来
        Parameters: pre_layer - 前一层网络，同为Layer类
        Return: 无
        """
        if pre_layer is None:
            assert self.input_shape is not None #如果没有前一层，则断言输入形状不为None，即为初始输入
            input_shape = self.input_shape #初始输入数据的尺寸形状
        else:
            input_shape = pre_layer.out_shape #本层输入数据的尺寸形状，即上一层的输出形状
        assert len(input_shape)==4 #断言输入数据的形状只有4维，数据个数，通道数，长和宽
        nb_batch,pre_nb_filter,pre_height,pre_width = input_shape  #取出数据个数，通道数(上一层滤波器个数),输入数据高度、宽度
        filter_height,filter_width = self.filter_size #取出卷积滤波器的高度和宽度
        height = (pre_height-filter_height)//self.stride +1 #卷积滤波后新的图像高度,这样可以保证新图像最后一个点是由一个完整的滤波器滤波结果得到
        width = (pre_width-filter_width)//self.stride +1 #卷积滤波后新的图像宽度
        self.out_shape = (nb_batch,self.nb_filter,height,width) #得出该层卷积滤波后的输出尺寸形状
        self.W = self.init((self.nb_filter,pre_nb_filter,filter_height,filter_width))  #随机初始化本卷积层权重参数矩阵
        self.b = Initial.get('zero')((self.nb_filter,)) #零值初始化截距参数矩阵
        
    def forward(self,input,*args,**kwargs):
        """
        Function:重构前向传递函数
        Parameters: input - 本层输入，上一层的输出
                    *args - 多余的非关键字参数存为元组传入
                    **kwargs - 多余的关键字参数存为字典传入
        Return: 本层的输出
        """
        self.last_input = input  #将输入存为上一次本层的输入（便于后续梯度计算）
        nb_batch, input_depth, old_img_h, old_img_w =input.shape #取出输入输入数据的尺寸，batch批次大小，图像深度，高度，宽度
        filter_h, filter_w = self.filter_size #取出滤波器高度和宽度
        new_img_h, new_img_w = self.out_shape[2:] #取出滤波后图像高度和宽度
        outputs = Initial.get('zero')((nb_batch,self.nb_filter,new_img_h,new_img_w))  #零值初始化输出数据矩阵
        for data_I in np.arange(nb_batch): #遍历每一个数据样本
            for filter_I in np.arange(self.nb_filter): #遍历每一个滤波器
                for h in np.arange(new_img_h): #遍历图像新的高度
                    for w in np.arange(new_img_w): #遍历图像新的宽度
                        h_shift, w_shift = h*self.stride, w*self.stride #定位新图像该点的原图像滤波器作用范围初始点
                        patch = input[data_I,:,h_shift:h_shift+filter_h,w_shift:w_shift+filter_w]  #定位得到新图像该点的原图像上滤波器作用范围
                        outputs[data_I,filter_I,h,w]=np.sum(patch*self.W[filter_I])+self.b[filter_I] #得到新图像该点的卷积结果
        self.last_output = self.activation.forward(outputs) #使用激活函数对卷积结果进一步过滤，得到最后的计算结果，并保存到上一次本层的输出(便于后续梯度计算)
        return self.last_output #返回前向传递计算结果
    
    def backward(self,pre_grad,*args,**kwargs):
        """
        Function:重构梯度反向传播函数
        Parameters: pre_grad - 后面一层的梯度
                    *args - 多余的非关键字参数存为元组传入
                    **kwargs - 多余的关键字参数存为字典传入
        Return: 本层的向前面一层传递的梯度
        """
        assert pre_grad.shape == self.last_output.shape #先断言后面一层传递的梯度尺寸形状是否与本层输出的尺寸形状匹配
        nb_batch, input_depth, old_img_h, old_img_w = self.last_input.shape #取出本层输入的批次，通道深度，原图像高度、宽度
        new_img_h, new_img_w =self.out_shape[2:] #取出本层输出的新的图像高度和宽度
        filter_h, filter_w = self.filter_size  #取出滤波器的高度和宽度
        self.dW = Initial.get('zero')(self.W.shape) #零值初始化权重参数矩阵梯度
        self.db = Initial.get('zero')(self.b.shape) #零值初始化截距梯度
        delta = pre_grad*self.activation.derivative() #后面一层梯度乘以激活函数的梯度，反向传递到激活函数之前的输出
        
        #计算dW
        for filter_I in np.arange(self.nb_filter): #遍历每个滤波器
            for depth_I in np.arange(input_depth): #遍历每个通道
                for h in np.arange(filter_h): #遍历滤波器高度
                    for w in np.arange(filter_w): #遍历滤波器宽度
                        input_window = self.last_input[:,depth_I,
                                                       h:old_img_h-filter_h+h+1:self.stride,
                                                       w:old_img_w-filter_w+w+1:self.stride] #定位该权重参数点在原图像上的作用范围，以高度为例，作用的第一个点为h，作用的最后一个点不超过old_h-filter_h+h
                        delta_window = delta[:,filter_I] #通过该点权重参数传递到输出的范围
                        self.dW[filter_I,depth_I,h,w]= np.sum(input_window*delta_window)/nb_batch #由已经传递的导数delta_window乘所作用的数据input_widow再求和（卷积）得到该权重参数的微分
        
        #计算db
        for filter_I in np.arange(self.nb_filter): #遍历每一个滤波器
            self.db[filter_I] = np.sum(delta[:,filter_I])/nb_batch  #每个滤波器的截距作用于卷积后的结果，由已经传递的导数delta与1卷积可得到该截距导数
        
        #计算对输入的导数dx，即传递给前面一层的导数
        layer_grad = Initial.get('zero')(self.last_input.shape) #零值初始化本层的导数
        for depth_I in np.arange(input_depth): #遍历原图像每一个通道
            for data_I in np.arange(nb_batch): #遍历每一个数据样本
                for filter_I in np.arange(self.nb_filter): #遍历每一个滤波器
                    for h in np.arange(new_img_h): #遍历新图像高度
                        for w in np.arange(new_img_w): #遍历新图像宽度
                            h_shift, w_shift =h*self.stride, w*self.stride #定位新图像该点的原图像滤波器作用范围初始点
                            #对于新图像上点(data_I,filter_I,h,w)，其由原图像(data_I,:,h_shift:h_shift+,w_shift:w_shift+)与filter_I滤波器卷积得到，则可对原图像这些点的导数累加已经传递梯度与滤波器权重的乘积
                            layer_grad[data_I,depth_I,h_shift:h_shift+filter_h,w_shift:w_shift+filter_w]+=self.W[filter_I,depth_I]*delta[data_I,filter_I,h,w]
        
        return layer_grad #返回本层传递给前面一层的导数
    
    @property #将权重矩阵、截距矩阵作为属性调用
    def params(self):
        """
        Function: 返回本层权重矩阵，截距矩阵
        Parameters: 无
        Return: self.W - 本层权重矩阵
                self.b - 本层截距矩阵
        """
        return self.W, self.b #返回本层权重矩阵，截距矩阵
    @property #将权重导数、截距导数作为属性调用
    def grads(self):
        """
        Function:返回本层权重导数，截距导数
        Parameters: 无
        Return: self.dW - 本层权重导数
                self.db - 本层截距导数
        """
        return self.dW, self.db #返回本层权重导数、截距导数


class MeanPooling(Layer):
    """
    本类为平均池化层类
    """
    def __init__(self,pool_size):
        self.pool_size=pool_size #池化层的池化尺寸
        self.out_shape=None #池化层的输出形状
        self.input_shape = None #池化层的输入形状
    
    def connect_to(self,pre_layer):
        """
        Function:重构连接上层函数，将本层池化层与前面一层的输出连接
        Parameters: pre_layer -前面一个神经网络层，父类同为Layer
        Return: 无
        """
        assert 5>len(pre_layer.out_shape)>=3 #断言前面一层网络输出的形状尺寸为4维或者3维
        old_h, old_w = pre_layer.out_shape[-2:] #取出前面一层网络输出的形状，即本层输入原图像的高度宽度
        pool_h, pool_w = self.pool_size #取出本层的池化高度宽度
        new_h, new_w = old_h//pool_h, old_w//pool_w #计算池化后新图像的高度宽度
        self.out_shape = pre_layer.out_shape[:-2]+(new_h,new_w) #得到本池化层的输出尺寸形状
        
    def forward(self,input,*args,**kwargs):
        """
        Function:重构前向传递函数，通过本层输入得到本层输出
        Parameters: input - 本层输入数据，即前面一层网络的输出数据
                    *args - 多余的非关键字参数作为元组传入
                    **kwargs - 多余的关键字参数作为字典传入
        Return: output - 本层的输出
        """
        self.input_shape = input.shape #本层输入数据的尺寸形状
        pool_h,pool_w = self.pool_size #取出池化核的高度宽度
        new_h,new_w = self.out_shape[-2:] #取出本层输出图像的高度宽度
        output = Initial.get('zero')(self.out_shape) #零值初始化本层的输出
        if np.ndim(input)==4:  #如果输入为4维，判断为卷积图像数据
            nb_batch, input_depth, old_h, old_w =input.shape #取出数据样本个数，原图像通道深度，高度，宽度
            for data_I in np.arange(nb_batch): #遍历每一个数据样本
                for depth_I in np.arange(input_depth): #遍历每一个通道深度
                    for h in np.arange(new_h): #遍历新图像高度
                        for w in np.arange(new_w): #遍历新图像宽度
                            h_shift, w_shift = h*pool_h, w*pool_w #求取(h,w)在原图像上的作用范围初始点
                            #新图像的[data_I,depth_I,h,w]的值为其作用范围的平均值
                            output[data_I,depth_I,h,w]=np.mean(input[data_I,depth_I,h_shift:h_shift+pool_h,w_shift:w_shift+pool_w])
            return output  #返回本层的输出
        elif np.ndim(input)==3:  #如果输入为3维，判断为循环序列数据
            nb_batch, old_h, old_w = input.shape #取出数据样本个数，数据高度(时间长度),宽度(单个时刻输出长度)
            for data_I in np.arange(nb_batch): #遍历样本数据
                for h in np.arange(new_h): #遍历池化后高度
                    for w in np.arange(new_w): #遍历池化后宽度
                        h_shift, w_shift = h*pool_h, w*pool_w #求取(h,w)在原图像上的作用范围初始点
                        #新图像的[data_I,h,w]的值为其作用范围的平均值
                        output[data_I,h,w]=np.mean(input[data_I,h_shift:h_shift+pool_h,w_shift:w_shift+pool_w])
            return output #返回本层输出
        else:
            raise ValueError() #输入数据维度有误
                        
    def backward(self,pre_grad,*args,**kwargs):
        """
        Function:重构梯度反向传播函数
        Parameters: pre_grad - 由后面一层传递过来的梯度
                    *args - 多余的非关键字参数存为元组传递
                    **kwargs - 多余的关键字参数存为字典传递
        Return: layer_grad - 本层往前传递的梯度
        """
        assert pre_grad.shape==self.out_shape #断言后面一层传递的梯度与本层输出尺寸形状一致
        pool_h, pool_w = self.pool_size #取出池化核的高度宽度
        kernel_ratio = 1.0/np.prod(self.pool_size) #由于是平均池化，则每一点的权重为1.0/池化核点数
        layer_grad = Initial.get('zero')(self.input_shape) #零值初始化本层往前面一层传递的梯度
        if np.ndim(pre_grad)==4: #传递梯度为4维，输入数据为卷积图像数据
            nb_batch, input_depth, new_h, new_w = pre_grad.shape #取出样本数据个数，图像通道深度，输出图像高度宽度
            for data_I in np.arange(nb_batch): #遍历每一个数据样本
                for depth_I in np.arange(input_depth): #遍历每一个深度
                    for h in np.arange(new_h): #遍历输出图像高度
                        for w in np.arange(new_w): #遍历输出图像宽度
                            h_shift, w_shift = h*pool_h, w*pool_w #求取(h,w)在输入图像上的作用范围初始点
                            #pre_grad(data_I,depth_I,h,w)作用的范围为(data_I,depth_I,h_shift:h_shift+,w_shift:w_shift+)
                            layer_grad[data_I,depth_I,h_shift:h_shift+pool_h,w_shift:w_shift+pool_w] = pre_grad[data_I,depth_I,h,w]*kernel_ratio
            return layer_grad  #返回本层对输入数据的梯度矩阵
        elif np.ndim(pre_grad)==3: #传递梯度为3维，输入数据为循环序列数据
            nb_batch, new_h, new_w = pre_grad.shape #取出样本数据个数，数据高度(时间长度),宽度(单个时刻输出长度)
            for data_I in np.arange(nb_batch): #遍历每一个数据样本
                for h in np.arange(new_h): #遍历输出数据高度
                    for w in np.arange(new_w): #遍历输出数据宽度
                        h_shift, w_shift = h*pool_h, w*pool_w #求取(h,w)在输入数据上的作用范围初始点
                        #pre_grad(data_I,h,w)作用的范围为(data_I,h_shift:h_shift+,w_shift:w_shift+)
                        layer_grad[data_I,h_shift:h_shift+pool_h,w_shift:w_shift+pool_w] = pre_grad[data_I,h,w]*kernel_ratio
            return layer_grad  #返回本层对输入数据的梯度矩阵
        else:
            raise ValueError() #输入数据维度有误
    
class Flatten(Layer):
    """
    构建Flatten层，对一个样本数据，将其转化成一维数据
    """
    def __init__(self,outdim=2):
        self.outdim = outdim #本层输出数据的维度，默认为2维，一维为数据样本个数，另一维为平坦后的一维单个数据
        self.input_shape = None #本层的输入数据的尺寸形状
        self.out_shape = None #本层的输出数据的尺寸形状
        
    def connect_to(self,pre_layer):
        """
        Function:重构连接前面一层函数，将本层Flatten层与前面一层连接
        Parameters: pre_layer - 前面一层神经网络，父类同为Layer
        Return: 无
        """
        assert 5>len(pre_layer.out_shape)>=3 #断言前面一层的输出数据形状为4维或3维
        self.input_shape = pre_layer.out_shape #前面一层的输出作为本层的输入
        to_flatten = np.prod(pre_layer.out_shape[self.outdim-1:]) #将索引为outdim-1开始的维度进行乘积，以便平坦化
        self.out_shape = pre_layer.out_shape[:self.outdim-1] +(to_flatten,)  #本层的输出数据形状尺寸，第一维为样本个数，第二维为平坦化的数据

    def forward(self,input,*args,**kwargs):
        """
        Function:重构前向传递函数，通过本层输入得到本层输出
        Parameters: input - 本层输入，即前面一层的输出数据
                    *args - 多余的非字典参数存为元组传递
                    **kwargs - 多余的字典参数存为字典传递
        Return:  本层的输出数据矩阵
        """
        assert self.input_shape == input.shape #断言输入数据形状与本层之前设定输入数据形状一致
        return np.reshape(input,self.out_shape) #变换输入数据的形状便可实现平坦化,返回本层的输出矩阵
    
    def backward(self,pre_grad,*args,**kwargs):
        """
        Function:重构梯度反向传播函数
        Parameters: pre_grad - 由后面一层传递到本层的梯度
                    *args - 多余的非字典参数存为元组传递
                    **kwargs -多余的字典参数存为字典传递
        Return: 本层传递到前面一层的梯度矩阵
        """
        #由于只是实现平坦化过程，本层输入输出矩阵数据一一对应且一致，即梯度全为1，只需将传递到本层的梯度进行reshape即可
        return np.reshape(pre_grad,self.input_shape) #返回本层传递到前一层的梯度

class Softmax(Layer):
    """
    本类为全连接层+softmax输出类
    """
    def __init__(self,n_out,n_in=None, init='glorot_uniform',activation='softmax'):
        self.n_out = n_out #本层某个样本数据输出的一维数据个数，即可能被分类的类别数
        self.n_in = n_in #本层某个样本数据输入的一维数据个数
        self.out_shape = (None,n_out) #本层的输出形状，第一维为样本数据遍历，第二维为某个样本的输出结果
        self.init = Initial.get(init) #得到本层的参数初始化类
        self.act_layer = Activation.get(activation) #得到本层的激活函数，即softmax函数
        self.W, self.dW =None, None #初始化本层的全连接权重参数及微分
        self.b, self.db = None, None #初始化本层的全连接截距和微分
        self.last_input = None #初始化本层的上一次输入
        
    def connect_to(self,pre_layer=None):
        """
        Function:重构连接前面一层神经网络函数
        Parameters: pre_layer - 前面一层神经网络，与本层父类同为Layer
        Return: 无
        """
        assert len(pre_layer.out_shape)==2 #断言前面一层输出数据维数为2，即前面一层为平坦层
        n_in = pre_layer.out_shape[-1] #取出前面一层输出单个数据的属性个数
        self.W = self.init((n_in,self.n_out)) #初始化本层的全连接权重
        self.b = Initial.get('zero')((self.n_out,)) #零值初始化本层的全连接截距
        
    def forward(self,input,*args,**kwargs):
        """
        Function: 重构前向传递函数，将前一层输出作为输入计算本层输出
        Parameters: input - 本层的输入，即前面一层的输出数据矩阵
        Return: output - 本层的输出，即各个类别的概率
        """
        self.last_input = input #将输入保存到本类的last_input，以便后续梯度计算
        allLink_output = np.dot(input,self.W)+self.b # 计算本层的全连接输出
        output = self.act_layer.forward(allLink_output) #通过激活函数计算本层的输出
        return output #返回本层的输出，softmax输出
    
    def backward(self,pre_grad,*args,**kwargs):
        """
        Function: 重构梯度反向传递函数
        Parameters: pre_grad - 由后面一层神经网络传递到本层输出的梯度矩阵
                    *args - 多余的非关键字参数存为元组传递
                    **kwargs - 多余的关键字参数存为字典传递
        Return: 本层对输入数据的梯度矩阵
        """
        act_grad = pre_grad*self.act_layer.derivative() #计算反向经过激活层传递到全连接输出的梯度
        nb_batch = act_grad.shape[0] #取出这一批次的数据样本个数
        #计算全连接权重微分，一个out与一个in连接，传递梯度与数据属性相乘为权重，最后求所有数据的权重微分平均
        self.dW = np.dot(self.last_input.T,act_grad)/nb_batch 
        #计算全连接截距微分,每个数据对所有截距梯度为1，则直接用已经传递的梯度，再求所有数据的截距微分平均
        self.db = np.mean(act_grad,axis=0)
        return np.dot(act_grad,self.W.T) #每个属性通过n_out个权重作用与全连接输出，反过来求取每个数据每个属性的梯度
    
    @property #将权重矩阵、截距矩阵作为属性调用
    def params(self):
        """
        Function: 返回本层权重矩阵，截距矩阵
        Parameters: 无
        Return: self.W - 本层权重矩阵
                self.b - 本层截距矩阵
        """
        return self.W, self.b #返回本层权重矩阵，截距矩阵
    
    @property #将权重导数、截距导数作为属性调用
    def grads(self):
        """
        Function:返回本层权重导数，截距导数
        Parameters: 无
        Return: self.dW - 本层权重导数
                self.db - 本层截距导数
        """
        return self.dW, self.db #返回本层权重导数、截距导数
        
class SimpleRNN(Layer):
    """
    本类为基本RNN网络层类，用于处理序列数据
    """
    def __init__(self,n_out,n_in=None,nb_batch=None,nb_seq=None,
                init='glorot_uniform',inner_init='orthogonal',
                activation='tanh',return_sequence=False):
        self.n_out = n_out #隐藏层的一维输出数据元素个数
        self.n_in = n_in #输入数据元素个数
        self.nb_batch = nb_batch #一批次的数据个数
        self.nb_seq = nb_seq #单个样本数据的序列数据个数
        self.init = Initial.get(init) #连接输入数据的参数矩阵初始化类
        self.inner_init = Initial.get(inner_init) #连接前隐藏层数据的参数矩阵初始化类
        self.activation_name = activation #隐藏层的激活函数类名
        self.activation = [] #空列表初始化隐藏层激活函数序列
        self.return_sequence = return_sequence #是否要输出所有序列数据前向传递结果的标志
        self.out_shape = None #初始化本层输出数据的形状
        self.last_input = None #用于保存本层输入数据
        self.last_output = None #用于保存本层输出数据
        self.W, self.dW = None, None #用于连接前隐藏层数据的权重矩阵及微分
        self.U, self.dU = None, None #用于连接序列输入数据的权重矩阵及微分
        self.b, self.db = None, None #隐藏层的偏倚及微分
        
    def connect_to(self,pre_layer=None):
        """
        Function: 重构连接前面一层神经网络函数
        Parameters: pre_layer - 前面一层神经网络，父类同为Layer类
        Return: 无
        """
        if pre_layer is not None:
            assert len(pre_layer.out_shape)==3 #如果pre_layer存在，断言其输出数据维度为3
            self.nb_batch = pre_layer.out_shape[0] #第一维度数据为该批次数据个数
            self.nb_seq = pre_layer.out_shape[1] #第二维数据为数据的序列数据个数
            self.n_in = pre_layer.out_shape[2] #第三维数据为单个序列数据的数据元素或属性个数
        else:
            assert self.n_in is not None #否则断言有数据输入
        if self.return_sequence:
            self.out_shape = (self.nb_batch,self.nb_seq,self.n_out)#输出所有序列数据预测结果
        else:
            self.out_shape = (self.nb_batch,self.n_out) #输出序列数据最后预测结果
        self.W = self.inner_init((self.n_out, self.n_out)) #随机初始化连接前隐藏层权重矩阵
        self.U = self.init((self.n_in,self.n_out)) #随机初始化连接输入数据的权重矩阵
        self.b = Initial.get('zero')((self.n_out)) #零值初始化隐藏层偏倚
        
    def forward(self,input,*args,**kwargs):
        """
        Function: 重构前向传递函数，计算本层输出
        Parameters: input - 本层数据输入，即前面一层网络输出数据
                    *args - 多余的非字典参数存为元组传递
                    **kwargs - 多余的字典参数存为字典传递
        Return: output - 本层输出
        """
        assert input.shape == (self.nb_batch,self.nb_seq,self.n_in) #断言输入数据与连接函数一样
        self.last_input = input #将输入存到本层上一次输入
        output = Initial.get('zero')((self.nb_batch,self.nb_seq,self.n_out)) #零值初始化本层输出数据矩阵
        if len(self.activation)==0: #如果本层的激活函数序列为空，添加序列激活函数类
            self.activation =[Activation.get(self.activation_name) for _ in np.arange(self.nb_seq) ]
        output[:,0,:] = self.activation[0].forward(np.dot(input[:,0,:],self.U)+self.b) #计算t=0时刻的隐藏层输出
        for time_I in np.arange(1,self.nb_seq): #遍历时间序列
            output[:,time_I,:]=self.activation[time_I].forward(
                    np.dot(input[:,time_I,:],self.U)+np.dot(output[:,time_I-1,:],self.W)+self.b) #计算t=time_I时刻的隐藏层输出
        self.last_output = output #将输出数据保存到last_output
        if self.return_sequence:
            return output #返回序列数据所有输出结果
        else:
            return output[:,-1,:] #返回序列数据最后输出结果
    
    def backward(self,pre_grad,*args,**kwargs):
        """
        Function:重构梯度反向传播函数，算法为BPTT，参考博客https://www.cnblogs.com/pinard/p/6509630.html
        Parameters: pre_grad - 由后面一层神经网络传递到本层的梯度
                    *args - 多余的非字典参数存为元组传递
                    **kwargs - 多余的字典参数存为字典传递
        Return: layer_grad - 本层神经网络向前面一层神经网络传递的梯度
        """
        assert pre_grad.shape == self.out_shape #断言由后面传递的梯度的形状是否与本层输出形状一致
        self.dW = Initial.get('zero')(self.W.shape) #零值初始化连接前隐藏层数据的权重微分
        self.dU = Initial.get('zero')(self.U.shape) #零值初始化连接输入数据的权重微分
        self.db = Initial.get('zero')(self.b.shape) #零值初始化隐藏层偏倚微分
        layer_grad = Initial.get('zero')((self.nb_batch,self.nb_seq,self.n_in)) #零值初始化对输入数据的梯度矩阵，即传递给前面一层的梯度矩阵
        if self.return_sequence:
            #返回的是序列数据所有输出结果
            delta_h = pre_grad[:,self.nb_seq-1,:]*self.activation[self.nb_seq-1].derivative() #传递到序列最后隐藏层激活前的梯度
            self.dW += np.dot(self.last_output[:,self.nb_seq-2,:].T,delta_h)/self.nb_batch #计算序列最后隐藏层dW
            self.dU += np.dot(self.last_input[:,self.nb_seq-1,:].T,delta_h)/self.nb_batch #计算序列最后隐藏层dU
            self.db += (np.sum(delta_h,axis=0)/self.nb_batch).reshape(self.b.shape) #计算序列最后隐藏层的db
            layer_grad[:,self.nb_seq-1,:]=np.dot(delta_h,self.U.T) #计算对序列最后时刻输入数据的导数
            for time_I in np.arange(self.nb_seq-2,0,-1): #方向遍历时间序列
                #传递到此时刻隐藏层的导数为该时刻输出此刻导数与后面时刻隐藏层往前传递的导数之和
                delta_h = (pre_grad[:,time_I,:]+np.dot(delta_h,self.W.T))*self.activation[time_I].derivative() #传递到此时刻隐藏层激活前的梯度
                self.dW += np.dot(self.last_output[:,time_I-1,:].T,delta_h)/self.nb_batch #计算序列此时刻隐藏层dW
                self.dU += np.dot(self.last_input[:,time_I,:].T,delta_h)/self.nb_batch #计算序列此时刻隐藏层dU
                self.db += (np.sum(delta_h,axis=0)/self.nb_batch).reshape(self.b.shape) #计算序列此时刻隐藏层的db
                layer_grad[:,time_I,:]=np.dot(delta_h,self.U.T) #计算对序列此时刻输入数据的导数
            delta_h=(pre_grad[:,0,:]+np.dot(delta_h,self.W.T))*self.activation[0].derivative() #传递到0时刻隐藏层激活前的梯度
            self.dU += np.dot(self.last_input[:,0,:].T,delta_h)/self.nb_batch #计算序列0时刻隐藏层dU
            self.db += (np.sum(delta_h,axis=0)/self.nb_batch).reshape(self.b.shape) #计算序列0时刻隐藏层的db
            layer_grad[:,0,:]=np.dot(delta_h,self.U.T) #计算对序列0时刻输入数据的导数
        else:
            #返回的是序列数据最后输出结果，那么在传递导数时没有此时刻(最后时刻除外)输出层的影响
            delta_h = pre_grad[:,0,:]*self.activation[self.nb_seq-1].derivative() #传递到序列最后隐藏层激活前的梯度
            self.dW += np.dot(self.last_output[:,self.nb_seq-2,:].T,delta_h)/self.nb_batch #计算序列最后隐藏层dW
            self.dU += np.dot(self.last_input[:,self.nb_seq-1,:].T,delta_h)/self.nb_batch #计算序列最后隐藏层dU
            self.db += (np.sum(delta_h,axis=0)/self.nb_batch).reshape(self.b.shape) #计算序列最后隐藏层的db
            layer_grad[:,self.nb_seq-1,:]=np.dot(delta_h,self.U.T) #计算对序列最后时刻输入数据的导数
            for time_I in np.arange(self.nb_seq-2,0,-1): #方向遍历时间序列
                #传递到此时刻隐藏层的导数为该时刻输出此刻导数与后面时刻隐藏层往前传递的导数之和
                delta_h = np.dot(delta_h,self.W.T)*self.activation[time_I].derivative() #传递到此时刻隐藏层激活前的梯度
                self.dW += np.dot(self.last_output[:,time_I-1,:].T,delta_h)/self.nb_batch #计算序列此时刻隐藏层dW
                self.dU += np.dot(self.last_input[:,time_I,:].T,delta_h)/self.nb_batch #计算序列此时刻隐藏层dU
                self.db += (np.sum(delta_h,axis=0)/self.nb_batch).reshape(self.b.shape) #计算序列此时刻隐藏层的db
                layer_grad[:,time_I,:]=np.dot(delta_h,self.U.T) #计算对序列此时刻输入数据的导数
            delta_h=np.dot(delta_h,self.W.T)*self.activation[0].derivative() #传递到0时刻隐藏层激活前的梯度
            self.dU += np.dot(self.last_input[:,0,:].T,delta_h)/self.nb_batch #计算序列0时刻隐藏层dU
            self.db += (np.sum(delta_h,axis=0)/self.nb_batch).reshape(self.b.shape) #计算序列0时刻隐藏层的db
            layer_grad[:,0,:]=np.dot(delta_h,self.U.T) #计算对序列0时刻输入数据的导数
        return layer_grad #返回本层对输入数据的梯度，即传递到前面一层神经网络的梯度
    
    @property #将权重矩阵、截距矩阵作为属性调用
    def params(self):
        """
        Function: 返回本层权重矩阵，截距矩阵
        Parameters: 无
        Return: self.W - 本层权重矩阵
                self.b - 本层截距矩阵
        """
        return self.W, self.U, self.b #返回本层权重矩阵，截距矩阵
    
    @property #将权重导数、截距导数作为属性调用
    def grads(self):
        """
        Function:返回本层权重导数，截距导数
        Parameters: 无
        Return: self.dW - 本层权重导数
                self.db - 本层截距导数
        """
        return self.dW, self.dU, self.db #返回本层权重导数、截距导数
        
class LSTM(Layer):
    """
    本类为基本LSTM网络层类，用于处理序列数据
    """
    def __init__(self,n_out,n_in=None,nb_batch=None,nb_seq=None,
                init='glorot_uniform',inner_init='orthogonal',
                activation_tan='tanh',activation_sig='sigmoid',return_sequence=False):
        self.n_out = n_out #隐藏层的一维输出数据元素个数
        self.n_in = n_in #输入数据元素个数
        self.nb_batch = nb_batch #一批次的数据个数
        self.nb_seq = nb_seq #单个样本数据的序列数据个数
        self.init = Initial.get(init) #连接输入数据的参数矩阵初始化类
        self.inner_init = Initial.get(inner_init) #连接前隐藏层数据的参数矩阵初始化类
        self.activation_tan = activation_tan #隐藏层的激活函数类名（tanh函数）
        self.activation_sig = activation_sig #隐藏层的激活函数类名(sigmoid函数)
        self.activation = [] #空列表初始化隐藏层激活函数序列
        self.return_sequence = return_sequence #是否要输出所有序列数据前向传递结果的标志
        self.out_shape = None #初始化本层输出数据的形状
        self.last_input = None #用于保存本层输入数据
        self.last_output = None #用于保存本层输出数据，ht序列数据
        self.last_ft = None #用于保存本层遗忘门ft序列数据
        self.last_it = None #用于保存本层输入门it序列数据
        self.last_at = None #用于保存本层输入门at序列数据
        self.last_ct = None #用于保存本层细胞状态ct序列数据
        self.last_ot = None #用于保存本层输出门ot序列数据
        self.Wf, self.dWf = None, None #用于连接前隐藏层数据的权重矩阵及微分,遗忘门f
        self.Uf, self.dUf = None, None #用于连接序列输入数据的权重矩阵及微分，遗忘门f
        self.bf, self.dbf = None, None #隐藏层的偏倚及微分，遗忘门f
        self.Wi, self.dWi = None, None #用于连接前隐藏层数据的权重矩阵及微分,输入门i
        self.Ui, self.dUi = None, None #用于连接序列输入数据的权重矩阵及微分，输入门i
        self.bi, self.dbi = None, None #隐藏层的偏倚及微分，输入门i
        self.Wa, self.dWa = None, None #用于连接前隐藏层数据的权重矩阵及微分,输入门a
        self.Ua, self.dUa = None, None #用于连接序列输入数据的权重矩阵及微分，输入门a
        self.ba, self.dba = None, None #隐藏层的偏倚及微分，输入门a
        self.Wo, self.dWo = None, None #用于连接前隐藏层数据的权重矩阵及微分,输出门o
        self.Uo, self.dUo = None, None #用于连接序列输入数据的权重矩阵及微分，输出门o
        self.bo, self.dbo = None, None #隐藏层的偏倚及微分，输出门o
        
    def connect_to(self,pre_layer=None):
        """
        Function:重构连接前面一层的神经网络函数
        Parameters: pre_layer - 前面一层神经网络，父类同为Layer类
        Return: 无
        """
        if pre_layer is not None:
            assert len(pre_layer.out_shape)==3 #如果pre_layer存在，断言其输出数据维度为3
            self.nb_batch = pre_layer.out_shape[0] #第一维数据为该批次数据个数
            self.nb_seq = pre_layer.out_shape[1] #第二维数据为数据的序列个数
            self.n_in = pre_layer.out_shape[2] #第三维数据为单个序列数据的数据元素或属性个数
        else:
            assert self.n_in is not None #否则断言有数据输入
        if self.return_sequence:
            self.out_shape = (self.nb_batch,self.nb_seq,self.n_out) #输出所有序列数据预测结果
        else:
            self.out_shape = (self.nb_batch,self.n_out) #输出序列数据最后预测结果
        self.Wf = self.inner_init((self.n_out,self.n_out)) #随机初始化连接前隐藏层权重矩阵，遗忘门f
        self.Uf = self.init((self.n_in,self.n_out)) #随机初始化连接输入数据的权重矩阵，遗忘门f
        self.bf = Initial.get('zero')((self.n_out))  #零值初始化隐藏层偏倚，遗忘门f
        self.Wi = self.inner_init((self.n_out,self.n_out)) #随机初始化连接前隐藏层权重矩阵，输入门i
        self.Ui = self.init((self.n_in,self.n_out)) #随机初始化连接输入数据的权重矩阵，输入门i
        self.bi = Initial.get('zero')((self.n_out))  #零值初始化隐藏层偏倚，输入门i
        self.Wa = self.inner_init((self.n_out,self.n_out)) #随机初始化连接前隐藏层权重矩阵，输入门a
        self.Ua = self.init((self.n_in,self.n_out)) #随机初始化连接输入数据的权重矩阵，输入门a
        self.ba = Initial.get('zero')((self.n_out))  #零值初始化隐藏层偏倚，输入门a
        self.Wo = self.inner_init((self.n_out,self.n_out)) #随机初始化连接前隐藏层权重矩阵，输出门o
        self.Uo = self.init((self.n_in,self.n_out)) #随机初始化连接输入数据的权重矩阵，输出门o
        self.bo = Initial.get('zero')((self.n_out))  #零值初始化隐藏层偏倚，输出门o
        
    def forward(self,input,*args,**kwargs):
        """
        Function:重构前向传递函数，计算本层输出
        Parameters: input - 本层数据输入，即前面一层网络输出数据
                    *args - 多余的非字典参数存为元组传递
                    **kwargs - 多余的字典参数存为字典传递
        Return: output - 本层输出
        """
        assert input.shape == (self.nb_batch,self.nb_seq,self.n_in) #断言输入数据与连接函数一样
        self.last_input = input #将输入保存到本层上一次输入
        output = Initial.get('zero')((self.nb_batch,self.nb_seq,self.n_out)) #零值初始化本层输出数据矩阵
        self.last_ft = Initial.get('zero')((self.nb_batch,self.nb_seq,self.n_out)) #零值初始化本层遗忘门ft序列数据
        self.last_it = Initial.get('zero')((self.nb_batch,self.nb_seq,self.n_out)) #零值初始化本层输入门it序列数据
        self.last_at = Initial.get('zero')((self.nb_batch,self.nb_seq,self.n_out)) #零值初始化本层输入门at序列数据
        self.last_ct = Initial.get('zero')((self.nb_batch,self.nb_seq,self.n_out)) #零值初始化本层细胞状态ct序列数据
        self.last_ot = Initial.get('zero')((self.nb_batch,self.nb_seq,self.n_out)) #零值初始化本层本层输出门ot序列数据
        if len(self.activation)==0: #如果本层的激活函数序列为空，添加序列激活函数类
            self.activation =[[Activation.get(self.activation_sig),Activation.get(self.activation_sig),Activation.get(self.activation_tan),
                               Activation.get(self.activation_sig),Activation.get(self.activation_tan)] for _ in np.arange(self.nb_seq)]
        #计算t=0时刻的隐藏层输出和细胞状态
        #遗忘门
        ft = self.activation[0][0].forward(np.dot(input[:,0,:],self.Uf)+self.bf) 
        #输入门
        it = self.activation[0][1].forward(np.dot(input[:,0,:],self.Ui)+self.bi)
        at = self.activation[0][2].forward(np.dot(input[:,0,:],self.Ua)+self.ba)
        #更新细胞状态
        ct = it*at
        #更新输出门输出
        ot = self.activation[0][3].forward(np.dot(input[:,0,:],self.Uo)+self.bo)
        ht = ot*self.activation[0][4].forward(ct)
        #保存0时刻的输出到output
        output[:,0,:]=ht
        self.last_ft[:,0,:]=ft #保存此时刻ft
        self.last_it[:,0,:]=it #保存此时刻it
        self.last_at[:,0,:]=at #保存此时刻at
        self.last_ct[:,0,:]=ct #保存此时刻ct
        self.last_ot[:,0,:]=ot #保存此时刻ot
        #遍历时间序列,计算t=time_I时刻的隐藏层输出和细胞状态
        for time_I in np.arange(1,self.nb_seq):
            #遗忘门
            ft = self.activation[time_I][0].forward(np.dot(ht,self.Wf)+np.dot(input[:,time_I,:],self.Uf)+self.bf)
            #输入门
            it = self.activation[time_I][1].forward(np.dot(ht,self.Wi)+np.dot(input[:,time_I,:],self.Ui)+self.bi)
            at = self.activation[time_I][2].forward(np.dot(ht,self.Wa)+np.dot(input[:,time_I,:],self.Ua)+self.ba)
            #更新细胞状态
            ct = ct*ft+it*at
            #更新输出门输出
            ot = self.activation[time_I][3].forward(np.dot(ht,self.Wo)+np.dot(input[:,time_I,:],self.Uo)+self.bo)
            ht = ot*self.activation[time_I][4].forward(ct)
            #保存time_I时刻的输出到output
            output[:,time_I,:]=ht
            self.last_ft[:,time_I,:]=ft #保存此时刻ft
            self.last_it[:,time_I,:]=it #保存此时刻it
            self.last_at[:,time_I,:]=at #保存此时刻at
            self.last_ct[:,time_I,:]=ct #保存此时刻ct
            self.last_ot[:,time_I,:]=ot #保存此时刻ot
        self.last_output = output #将输出结果保存到last_output
        if self.return_sequence:
            return output  #返回序列数据所有输出结果
        else:
            return output[:,-1,:] #返回序列数据最后输出结果
    
    def backward(self,pre_grad,*args,**kwargs):
        """
        Function:重构梯度反向传播函数，算法为BPTT，参考博客https://www.cnblogs.com/pinard/p/6519110.html
        本梯度反向传播与博客推导有所不同，关键点是反向传递ct的导数，以及反向传递到ht的导数计算
        Parameters: pre_grad - 由后面一层神经网络传递到本层的梯度
                    *args - 多余的非字典参数存为元组传递
                    **kwargs - 多余的字典参数存为字典传递
        Return: layer_grad - 本层神经网络向前面一层神经网络传递的梯度
        """
        assert pre_grad.shape == self.out_shape  #断言由后面传递的梯度的形状是否与本层输出形状一致
        self.dWf = Initial.get('zero')(self.Wf.shape) #零值初始化连接前隐藏层数据的权重微分，遗忘门f
        self.dUf = Initial.get('zero')(self.Uf.shape) #零值初始化连接输入数据的权重微分，遗忘门f
        self.dbf = Initial.get('zero')(self.bf.shape) #零值初始化隐藏层偏倚微分，遗忘门f
        self.dWi = Initial.get('zero')(self.Wi.shape) #零值初始化连接前隐藏层数据的权重微分，输入门i
        self.dUi = Initial.get('zero')(self.Ui.shape) #零值初始化连接输入数据的权重微分，输入门i
        self.dbi = Initial.get('zero')(self.bi.shape) #零值初始化隐藏层偏倚微分，输入门i
        self.dWa = Initial.get('zero')(self.Wa.shape) #零值初始化连接前隐藏层数据的权重微分，输入门a
        self.dUa = Initial.get('zero')(self.Ua.shape) #零值初始化连接输入数据的权重微分，输入门a 
        self.dba = Initial.get('zero')(self.ba.shape) #零值初始化隐藏层偏倚微分，输入门a
        self.dWo = Initial.get('zero')(self.Wo.shape) #零值初始化连接前隐藏层数据的权重微分，输出门o
        self.dUo = Initial.get('zero')(self.Uo.shape) #零值初始化连接输入数据的权重微分，输出门o
        self.dbo = Initial.get('zero')(self.bo.shape) #零值初始化隐藏层偏倚微分，输出门o
        layer_grad = Initial.get('zero')((self.nb_batch,self.nb_seq,self.n_in)) #零值初始化对输入数据的梯度矩阵，即传递给前面一层的梯度矩阵
        if self.return_sequence:
            #返回的是序列数据所有输出结果
            #传递到最后时刻输出ht的梯度
            delta_h = pre_grad[:,self.nb_seq-1,:]
            #传递到最后时刻细胞状态ct的梯度
            delta_c = delta_h*self.last_ot[:,self.nb_seq-1,:]*self.activation[self.nb_seq-1][4].derivative()
            #传递到遗忘门ft的导数
            delta_ft = delta_c* self.last_ct[:,self.nb_seq-2,:]*self.activation[self.nb_seq-1][0].derivative()
            #传递到输入门it的导数
            delta_it = delta_c*self.last_at[:,self.nb_seq-1,:]*self.activation[self.nb_seq-1][1].derivative()
            #传递到输入门at的导数
            delta_at = delta_c*self.last_it[:,self.nb_seq-1,:]*self.activation[self.nb_seq-1][2].derivative()
            #传递到输出门ot的导数
            delta_ot = delta_h*np.tanh(self.last_ct[:,self.nb_seq-1,:])*self.activation[self.nb_seq-1][3].derivative()
            #计算各权重和偏倚的微分
            self.dWf += np.dot(self.last_output[:,self.nb_seq-2,:].T,delta_ft)/self.nb_batch #计算序列最后时刻dWf
            self.dUf += np.dot(self.last_input[:,self.nb_seq-1,:].T,delta_ft)/self.nb_batch #计算序列最后时刻dUf
            self.dbf += (np.sum(delta_ft,axis=0)/self.nb_batch).reshape(self.bf.shape) #计算序列最后时刻dbf
            self.dWi += np.dot(self.last_output[:,self.nb_seq-2,:].T,delta_it)/self.nb_batch #计算序列最后时刻dWi
            self.dUi += np.dot(self.last_input[:,self.nb_seq-1,:].T,delta_it)/self.nb_batch #计算序列最后时刻dUi
            self.dbi += (np.sum(delta_it,axis=0)/self.nb_batch).reshape(self.bi.shape) #计算序列最后时刻dbf
            self.dWa += np.dot(self.last_output[:,self.nb_seq-2,:].T,delta_at)/self.nb_batch #计算序列最后时刻dWa
            self.dUa += np.dot(self.last_input[:,self.nb_seq-1,:].T,delta_at)/self.nb_batch #计算序列最后时刻dUa
            self.dba += (np.sum(delta_at,axis=0)/self.nb_batch).reshape(self.ba.shape) #计算序列最后时刻dba
            self.dWo += np.dot(self.last_output[:,self.nb_seq-2,:].T,delta_ot)/self.nb_batch #计算序列最后时刻dWo
            self.dUo += np.dot(self.last_input[:,self.nb_seq-1,:].T,delta_ot)/self.nb_batch #计算序列最后时刻dUo
            self.dbo += (np.sum(delta_ot,axis=0)/self.nb_batch).reshape(self.bo.shape) #计算序列最后时刻dbo
            #计算对输入数据的微分
            layer_grad[:,self.nb_seq-1,:]=np.dot(delta_ft,self.Uf.T)+np.dot(delta_it,self.Ui.T)+np.dot(delta_at,self.Ua.T)+np.dot(delta_ot,self.Uo.T)
            #传递到下一层ht-1的微分
            delta_ht_1 = np.dot(delta_ft,self.Wf.T)+np.dot(delta_it,self.Wi.T)+np.dot(delta_at,self.Wa.T)+np.dot(delta_ot,self.Wo.T)
            #传递到下一层ct-1的微分
            delta_ct_1 = delta_c*self.last_ft[:,self.nb_seq-1,:]
            #反向遍历nb_seq-2到1时刻
            for time_I in np.arange(self.nb_seq-2,0,-1):
                #传递到时刻time_I输出ht的梯度
                delta_h = pre_grad[:,time_I,:]+delta_ht_1
                #传递到时刻time_I细胞状态ct的梯度
                delta_c = delta_h*self.last_ot[:,time_I,:]*self.activation[time_I][4].derivative()+delta_ct_1
                #传递到遗忘门ft的导数
                delta_ft = delta_c* self.last_ct[:,time_I-1,:]*self.activation[time_I][0].derivative()
                #传递到输入门it的导数
                delta_it = delta_c*self.last_at[:,time_I,:]*self.activation[time_I][1].derivative()
                #传递到输入门at的导数
                delta_at = delta_c*self.last_it[:,time_I,:]*self.activation[time_I][2].derivative()
                #传递到输出门ot的导数
                delta_ot = delta_h*np.tanh(self.last_ct[:,time_I,:])*self.activation[time_I][3].derivative()
                #计算各权重和偏倚的微分
                self.dWf += np.dot(self.last_output[:,time_I-1,:].T,delta_ft)/self.nb_batch #计算序列此时刻dWf
                self.dUf += np.dot(self.last_input[:,time_I,:].T,delta_ft)/self.nb_batch #计算序列此时刻dUf
                self.dbf += (np.sum(delta_ft,axis=0)/self.nb_batch).reshape(self.bf.shape) #计算序列此时刻dbf
                self.dWi += np.dot(self.last_output[:,time_I-1,:].T,delta_it)/self.nb_batch #计算序列此时刻dWi
                self.dUi += np.dot(self.last_input[:,time_I,:].T,delta_it)/self.nb_batch #计算序列此时刻dUi
                self.dbi += (np.sum(delta_it,axis=0)/self.nb_batch).reshape(self.bi.shape) #计算序列此时刻dbf
                self.dWa += np.dot(self.last_output[:,time_I-1,:].T,delta_at)/self.nb_batch #计算序列此时刻dWa
                self.dUa += np.dot(self.last_input[:,time_I,:].T,delta_at)/self.nb_batch #计算序列此时刻dUa
                self.dba += (np.sum(delta_at,axis=0)/self.nb_batch).reshape(self.ba.shape) #计算序列此时刻dba
                self.dWo += np.dot(self.last_output[:,time_I-1,:].T,delta_ot)/self.nb_batch #计算序列此时刻dWo
                self.dUo += np.dot(self.last_input[:,time_I,:].T,delta_ot)/self.nb_batch #计算序列此时刻dUo
                self.dbo += (np.sum(delta_ot,axis=0)/self.nb_batch).reshape(self.bo.shape) #计算序列此时刻dbo
                #计算对输入数据的微分
                layer_grad[:,time_I,:]=np.dot(delta_ft,self.Uf.T)+np.dot(delta_it,self.Ui.T)+np.dot(delta_at,self.Ua.T)+np.dot(delta_ot,self.Uo.T)
                #传递到下一层ht-1的微分
                delta_ht_1 = np.dot(delta_ft,self.Wf.T)+np.dot(delta_it,self.Wi.T)+np.dot(delta_at,self.Wa.T)+np.dot(delta_ot,self.Wo.T)
                #传递到下一层ct-1的微分
                delta_ct_1 = delta_c*self.last_ft[:,time_I,:]
            #计算传递到0时刻的导数
            #传递到时刻0输出ht的梯度
            delta_h = pre_grad[:,0,:]+delta_ht_1
            #传递到时刻0细胞状态ct的梯度
            delta_c = delta_h*self.last_ot[:,0,:]*self.activation[0][4].derivative()+delta_ct_1
            #传递到遗忘门ft的导数,0时刻没有遗忘门
            #delta_ft = delta_c* self.last_ct[:,time_I-1,:]*self.activation[time_I][0].derivative()
            #传递到输入门it的导数
            delta_it = delta_c*self.last_at[:,0,:]*self.activation[0][1].derivative()
            #传递到输入门at的导数
            delta_at = delta_c*self.last_it[:,0,:]*self.activation[0][2].derivative()
            #传递到输出门ot的导数
            delta_ot = delta_h*np.tanh(self.last_ct[:,0,:])*self.activation[0][3].derivative()
            #计算各权重和偏倚的微分
            #self.dWf += np.dot(self.last_output[:,time_I-1,:].T,delta_ft)/self.nb_batch #计算序列此时刻dWf
            #self.dUf += np.dot(self.last_input[:,time_I,:].T,delta_ft)/self.nb_batch #计算序列此时刻dUf
            #self.dbf += (np.sum(delta_ft,axis=0)/self.nb_batch).reshape(self.bf.shape) #计算序列此时刻dbf
            #self.dWi += np.dot(self.last_output[:,time_I-1,:].T,delta_it)/self.nb_batch #计算序列此时刻dWi
            self.dUi += np.dot(self.last_input[:,0,:].T,delta_it)/self.nb_batch #计算序列此时刻dUi
            self.dbi += (np.sum(delta_it,axis=0)/self.nb_batch).reshape(self.bi.shape) #计算序列此时刻dbf
            #self.dWa += np.dot(self.last_output[:,time_I-1,:].T,delta_at)/self.nb_batch #计算序列此时刻dWa
            self.dUa += np.dot(self.last_input[:,0,:].T,delta_at)/self.nb_batch #计算序列此时刻dUa
            self.dba += (np.sum(delta_at,axis=0)/self.nb_batch).reshape(self.ba.shape) #计算序列此时刻dba
            #self.dWo += np.dot(self.last_output[:,time_I-1,:].T,delta_ot)/self.nb_batch #计算序列此时刻dWo
            self.dUo += np.dot(self.last_input[:,0,:].T,delta_ot)/self.nb_batch #计算序列此时刻dUo
            self.dbo += (np.sum(delta_ot,axis=0)/self.nb_batch).reshape(self.bo.shape) #计算序列此时刻dbo
            #计算对输入数据的微分
            layer_grad[:,0,:]=np.dot(delta_it,self.Ui.T)+np.dot(delta_at,self.Ua.T)+np.dot(delta_ot,self.Uo.T)
        else:
            #返回的是序列数据最后输出结果，那么在传递导数时没有此时刻(最后时刻除外)输出层的影响
            #传递到最后时刻输出ht的梯度
            delta_h = pre_grad[:,0,:]
            #传递到最后时刻细胞状态ct的梯度
            delta_c = delta_h*self.last_ot[:,self.nb_seq-1,:]*self.activation[self.nb_seq-1][4].derivative()
            #传递到遗忘门ft的导数
            delta_ft = delta_c* self.last_ct[:,self.nb_seq-2,:]*self.activation[self.nb_seq-1][0].derivative()
            #传递到输入门it的导数
            delta_it = delta_c*self.last_at[:,self.nb_seq-1,:]*self.activation[self.nb_seq-1][1].derivative()
            #传递到输入门at的导数
            delta_at = delta_c*self.last_it[:,self.nb_seq-1,:]*self.activation[self.nb_seq-1][2].derivative()
            #传递到输出门ot的导数
            delta_ot = delta_h*np.tanh(self.last_ct[:,self.nb_seq-1,:])*self.activation[self.nb_seq-1][3].derivative()
            #计算各权重和偏倚的微分
            self.dWf += np.dot(self.last_output[:,self.nb_seq-2,:].T,delta_ft)/self.nb_batch #计算序列最后时刻dWf
            self.dUf += np.dot(self.last_input[:,self.nb_seq-1,:].T,delta_ft)/self.nb_batch #计算序列最后时刻dUf
            self.dbf += (np.sum(delta_ft,axis=0)/self.nb_batch).reshape(self.bf.shape) #计算序列最后时刻dbf
            self.dWi += np.dot(self.last_output[:,self.nb_seq-2,:].T,delta_it)/self.nb_batch #计算序列最后时刻dWi
            self.dUi += np.dot(self.last_input[:,self.nb_seq-1,:].T,delta_it)/self.nb_batch #计算序列最后时刻dUi
            self.dbi += (np.sum(delta_it,axis=0)/self.nb_batch).reshape(self.bi.shape) #计算序列最后时刻dbf
            self.dWa += np.dot(self.last_output[:,self.nb_seq-2,:].T,delta_at)/self.nb_batch #计算序列最后时刻dWa
            self.dUa += np.dot(self.last_input[:,self.nb_seq-1,:].T,delta_at)/self.nb_batch #计算序列最后时刻dUa
            self.dba += (np.sum(delta_at,axis=0)/self.nb_batch).reshape(self.ba.shape) #计算序列最后时刻dba
            self.dWo += np.dot(self.last_output[:,self.nb_seq-2,:].T,delta_ot)/self.nb_batch #计算序列最后时刻dWo
            self.dUo += np.dot(self.last_input[:,self.nb_seq-1,:].T,delta_ot)/self.nb_batch #计算序列最后时刻dUo
            self.dbo += (np.sum(delta_ot,axis=0)/self.nb_batch).reshape(self.bo.shape) #计算序列最后时刻dbo
            #计算对输入数据的微分
            layer_grad[:,self.nb_seq-1,:]=np.dot(delta_ft,self.Uf.T)+np.dot(delta_it,self.Ui.T)+np.dot(delta_at,self.Ua.T)+np.dot(delta_ot,self.Uo.T)
            #传递到下一层ht-1的微分
            delta_ht_1 = np.dot(delta_ft,self.Wf.T)+np.dot(delta_it,self.Wi.T)+np.dot(delta_at,self.Wa.T)+np.dot(delta_ot,self.Wo.T)
            #传递到下一层ct-1的微分
            delta_ct_1 = delta_c*self.last_ft[:,self.nb_seq-1,:]
            #反向遍历nb_seq-2到1时刻
            for time_I in np.arange(self.nb_seq-2,0,-1):
                #传递到时刻time_I输出ht的梯度
                delta_h = delta_ht_1
                #传递到时刻time_I细胞状态ct的梯度
                delta_c = delta_h*self.last_ot[:,time_I,:]*self.activation[time_I][4].derivative()+delta_ct_1
                #传递到遗忘门ft的导数
                delta_ft = delta_c* self.last_ct[:,time_I-1,:]*self.activation[time_I][0].derivative()
                #传递到输入门it的导数
                delta_it = delta_c*self.last_at[:,time_I,:]*self.activation[time_I][1].derivative()
                #传递到输入门at的导数
                delta_at = delta_c*self.last_it[:,time_I,:]*self.activation[time_I][2].derivative()
                #传递到输出门ot的导数
                delta_ot = delta_h*np.tanh(self.last_ct[:,time_I,:])*self.activation[time_I][3].derivative()
                #计算各权重和偏倚的微分
                self.dWf += np.dot(self.last_output[:,time_I-1,:].T,delta_ft)/self.nb_batch #计算序列此时刻dWf
                self.dUf += np.dot(self.last_input[:,time_I,:].T,delta_ft)/self.nb_batch #计算序列此时刻dUf
                self.dbf += (np.sum(delta_ft,axis=0)/self.nb_batch).reshape(self.bf.shape) #计算序列此时刻dbf
                self.dWi += np.dot(self.last_output[:,time_I-1,:].T,delta_it)/self.nb_batch #计算序列此时刻dWi
                self.dUi += np.dot(self.last_input[:,time_I,:].T,delta_it)/self.nb_batch #计算序列此时刻dUi
                self.dbi += (np.sum(delta_it,axis=0)/self.nb_batch).reshape(self.bi.shape) #计算序列此时刻dbf
                self.dWa += np.dot(self.last_output[:,time_I-1,:].T,delta_at)/self.nb_batch #计算序列此时刻dWa
                self.dUa += np.dot(self.last_input[:,time_I,:].T,delta_at)/self.nb_batch #计算序列此时刻dUa
                self.dba += (np.sum(delta_at,axis=0)/self.nb_batch).reshape(self.ba.shape) #计算序列此时刻dba
                self.dWo += np.dot(self.last_output[:,time_I-1,:].T,delta_ot)/self.nb_batch #计算序列此时刻dWo
                self.dUo += np.dot(self.last_input[:,time_I,:].T,delta_ot)/self.nb_batch #计算序列此时刻dUo
                self.dbo += (np.sum(delta_ot,axis=0)/self.nb_batch).reshape(self.bo.shape) #计算序列此时刻dbo
                #计算对输入数据的微分
                layer_grad[:,time_I,:]=np.dot(delta_ft,self.Uf.T)+np.dot(delta_it,self.Ui.T)+np.dot(delta_at,self.Ua.T)+np.dot(delta_ot,self.Uo.T)
                #传递到下一层ht-1的微分
                delta_ht_1 = np.dot(delta_ft,self.Wf.T)+np.dot(delta_it,self.Wi.T)+np.dot(delta_at,self.Wa.T)+np.dot(delta_ot,self.Wo.T)
                #传递到下一层ct-1的微分
                delta_ct_1 = delta_c*self.last_ft[:,time_I,:]
            #计算传递到0时刻的导数
            #传递到时刻0输出ht的梯度
            delta_h =delta_ht_1
            #传递到时刻0细胞状态ct的梯度
            delta_c = delta_h*self.last_ot[:,0,:]*self.activation[0][4].derivative()+delta_ct_1
            #传递到遗忘门ft的导数,0时刻没有遗忘门
            #delta_ft = delta_c* self.last_ct[:,time_I-1,:]*self.activation[time_I][0].derivative()
            #传递到输入门it的导数
            delta_it = delta_c*self.last_at[:,0,:]*self.activation[0][1].derivative()
            #传递到输入门at的导数
            delta_at = delta_c*self.last_it[:,0,:]*self.activation[0][2].derivative()
            #传递到输出门ot的导数
            delta_ot = delta_h*np.tanh(self.last_ct[:,0,:])*self.activation[0][3].derivative()
            #计算各权重和偏倚的微分
            #self.dWf += np.dot(self.last_output[:,time_I-1,:].T,delta_ft)/self.nb_batch #计算序列此时刻dWf
            #self.dUf += np.dot(self.last_input[:,time_I,:].T,delta_ft)/self.nb_batch #计算序列此时刻dUf
            #self.dbf += (np.sum(delta_ft,axis=0)/self.nb_batch).reshape(self.bf.shape) #计算序列此时刻dbf
            #self.dWi += np.dot(self.last_output[:,time_I-1,:].T,delta_it)/self.nb_batch #计算序列此时刻dWi
            self.dUi += np.dot(self.last_input[:,0,:].T,delta_it)/self.nb_batch #计算序列此时刻dUi
            self.dbi += (np.sum(delta_it,axis=0)/self.nb_batch).reshape(self.bi.shape) #计算序列此时刻dbf
            #self.dWa += np.dot(self.last_output[:,time_I-1,:].T,delta_at)/self.nb_batch #计算序列此时刻dWa
            self.dUa += np.dot(self.last_input[:,0,:].T,delta_at)/self.nb_batch #计算序列此时刻dUa
            self.dba += (np.sum(delta_at,axis=0)/self.nb_batch).reshape(self.ba.shape) #计算序列此时刻dba
            #self.dWo += np.dot(self.last_output[:,time_I-1,:].T,delta_ot)/self.nb_batch #计算序列此时刻dWo
            self.dUo += np.dot(self.last_input[:,0,:].T,delta_ot)/self.nb_batch #计算序列此时刻dUo
            self.dbo += (np.sum(delta_ot,axis=0)/self.nb_batch).reshape(self.bo.shape) #计算序列此时刻dbo
            #计算对输入数据的微分
            layer_grad[:,0,:]=np.dot(delta_it,self.Ui.T)+np.dot(delta_at,self.Ua.T)+np.dot(delta_ot,self.Uo.T)
        return layer_grad #返回本层对输入数据的梯度，即传递到前面一层神经网络的梯度
    
    @property #将权重矩阵、截距矩阵作为属性调用
    def params(self):
        """
        Function: 返回本层权重矩阵，截距矩阵
        Parameters: 无
        Return: self.W - 本层权重矩阵
                self.b - 本层截距矩阵
        """
        return self.Wf,self.Uf,self.bf,self.Wi,self.Ui,self.bi,self.Wa,self.Ua,self.ba,self.Wo,self.Uo,self.bo #返回本层权重矩阵，截距矩阵
    
    @property #将权重导数、截距导数作为属性调用
    def grads(self):
        """
        Function:返回本层权重导数，截距导数
        Parameters: 无
        Return: self.dW - 本层权重导数
                self.db - 本层截距导数
        """
        return self.dWf,self.dUf,self.dbf,self.dWi,self.dUi,self.dbi,self.dWa,self.dUa,self.dba,self.dWo,self.dUo,self.dbo #返回本层权重导数、截距导数

class Embedding(Layer):
    """
    定义嵌入Embedding层类，本质为全连接层，用于数据降维和特征表示
    """
    def __init__(self,input_size=None,n_out=None,nb_batch=None,nb_seq=None,init='uniform'):
        self.nb_batch = nb_batch #单批次样本数据个数
        self.nb_seq = nb_seq #单个样本序列数据个数
        self.embed_words = Initial.get(init)((input_size,n_out)) #初始化embed降维矩阵
        self.d_embed_words = None #embed降维矩阵微分
        self.last_input = None #保存本茨输入数据
        self.out_shape = None #本层网路输出数据形状
        
    def connect_to(self,pre_layer=None):
        """
        Function:重构网络连接函数，将本层网络与前面一层网络相连接
        Parameters: pre_layer - 前面一层神经网络，父类同为Layer
        Return: 无
        """
        #本层的输出数据形状
        self.out_shape = (self.nb_batch,self.nb_seq,self.embed_words.shape[1])
        
    def forward(self,input,*args,**kwargs):
        """
        Function:重构前向传递函数，计算本层输出
        Parameters: input - 本层数据输入，即前面一层网络输出或原始输入数据
                    *args - 多余的非字典参数作为元组传递
                    **kwargs - 多余的字典参数作为字典传递
        Return: 本层网络输出
        """
        assert np.ndim(input)==3 #断言输入数据为三维数据
        self.last_input = input #保存本次输入数据
        return np.dot(input,self.embed_words) #进行降维操作，返回本次输出数据
    
    def backward(self,pre_grad,*args,**kwargs):
        """
        Function: 重构梯度反向传播函数，计算本层的梯度
        Parameters: pre_grad - 由后面一层往前传递到本层的梯度
                    *args - 多余的非字典参数作为元组传递
                    **kwargs - 多余的字典参数作为字典传递
        Return: layer_grad - 本层对输入数据的梯度
        """
        assert pre_grad.shape == self.out_shape #断言传递梯度形状与本层输出数据形状一致
        #零值初始化降维矩阵权重微分
        self.d_embed_words = Initial.get('zero')(self.embed_words.shape)
        #零值初始化本层对输入数据的导数
        layer_grad = Initial.get('zero')((self.nb_batch,self.nb_seq,self.embed_words.shape[0]))
        for data_I in np.arange(self.nb_batch): #遍历每一个样本数据，也可以整体计算
            for time_I in np.arange(self.nb_seq): #遍历样本的序列数据
                data_Now = self.last_input[data_I,time_I,:] #取出单个输入数据
                grad_Now = pre_grad[data_I,time_I,:] #取出此数据对应的传递梯度
                #更新降维矩阵权重微分
                self.d_embed_words += np.dot(data_Now.reshape(-1,1),grad_Now.reshape(1,-1))
                #对此输入数据的微分
                layer_grad[data_I,time_I,:] = np.dot(self.embed_words,grad_Now)
        self.d_embed_words = self.d_embed_words/self.nb_batch #取样本数据的平均
        return layer_grad #本层对输入数据的微分
    
    @property #将降维矩阵权重作为属性调用
    def params(self):
        """
        Function: 返回本层降维矩阵权重
        Parameters: 无
        Return: self.embed_words - 本层降维矩阵权重
        """
        return self.embed_words #返回本层降维矩阵权重
    
    @property #将降维矩阵权重微分作为属性调用
    def grads(self):
        """
        Function:返回本层降维矩阵权重微分
        Parameters: 无
        Return: self.dW - 本层降维矩阵权重微分
        """
        return self.d_embed_words #返回本层降维矩阵权重微分
        
    
                
            
        
        
        
            
        

