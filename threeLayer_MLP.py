# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:42:16 2019
本程序实现单隐藏层的多层感知机全连接网络
隐藏单元激活函数为整流线性单元ReLU
网络输出采用softmax单元输出
代价函数选用负对数似然（交叉熵代价函数）
代价函数正则化选用L2参数正则化
训练方法为梯度下降法
参考资料Ian Goodfellow《深度学习》
https://github.com/jldbc/numpy_neural_net
@author: yanji
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

def ReLU(X):
    """
    Function: 输入X，计算整流线性单元激活函数的输出
    Parameters: X - 函数输入
    Return : 激活函数输出
    """
    return np.maximum(X,0)   #返回激活函数输出

def ReLU_derivative(X):
    """
    Function: 计算激活函数的导数
    Parameters: X - 激活函数输入
    Return: 激活函数导数
    """
    return 1.0*(X>0) #X>0时返回1，X<=0时返回0

def built_model(X,hidden_nodes,output_dim=2):
    """
    Function:依据输入维数，隐藏层节点数，输出维数初始化神经网络
    Parameters: X - 样本数据矩阵
                hidden_node - 隐藏层的数目
                output_dims - 输出维数
    Return: model - 含有所有连接的权重与各层偏置的字典
    """
    model={}  #新建模型的空字典
    input_dim=X.shape[1]  #  样本数据的属性维数     
    model['W1']=np.random.randn(input_dim,hidden_nodes)/np.sqrt(input_dim) #随机初始化输入层到隐藏层的权重矩阵，采用正态分布，方差为input_dim
    model['b1']=np.zeros((1,hidden_nodes)) #零值初始化隐藏层的偏置
    model['W2']=np.random.randn(hidden_nodes,output_dim)/np.sqrt(hidden_nodes) #随机初始化隐藏层到输出层的权重矩阵，采用正态分布，方程为隐藏层节点数
    model['b2']=np.zeros((1,output_dim)) #零值初始化输出层的偏置
    return model #返回初始化后的神经网络字典

def feed_forward(model,X):
    """
    Function: 前向传递，输入样本数据矩阵，计算各层输入输出
    Parameters: model - 神经网络模型
                x - 样本数据矩阵
    Return: z1 - 隐藏层输入
            a1 - 隐藏层输出
            z2 - 输出层输入
            out - 输出层输出
    """
    W1,b1,W2,b2=model['W1'],model['b1'],model['W2'],model['b2'] #导出神经网络的权重与偏置
    z1=np.dot(X,W1)+b1  #计算隐藏层的输入
    a1=ReLU(z1) #采用整流线性单元激活函数，计算隐藏层输出
    z2=np.dot(a1,W2)+b2 #计算输出单元的输入
    exp_scores=np.exp(z2) #对输出单元的输入进行指数化
    out=exp_scores/np.sum(exp_scores,axis=1,keepdims=True)  #输出单元采用softmax函数进行输出
    return z1,a1,z2,out # 返回隐藏层输入输出，输出层输入输出

def calculate_loss(model,X,y,reg_lambda):
    """
    Function: 计算代价函数，包括负对数似然与正则化项
    Parameters: model - 神经网络模型
                X - 样本数据矩阵
                y - 样本数据类别标签矩阵，在本程序中，使用datasets.make_moons的0/1二分类
                reg_lambda - 正则化参数
    """
    num_examples = X.shape[0] #样本数据个数
    W1,b1,W2,b2 = model['W1'],model['b1'],model['W2'],model['b2'] #导出神经网络的权重与偏置
    z1,a1,z2,out=feed_forward(model,X) #计算模型对样本数据矩阵预测的输出
    #为减少计算量，样本数据类别标签矩阵生成在训练前完成
    #labelMat=np.zeros((num_examples,out.shape[1])) #零值初始化样本真实类别标签
    #for num in range(num_examples):
    #    labelMat[num,y[num]]=1  #根据样本数据真实类别标签将相应类别标志为1
    cross_entropy = -np.sum(np.log(out)*y)/num_examples  #计算交叉熵,loss=-sum(yiLabel*log(softmax(yi)))
    loss=cross_entropy+reg_lambda/2*(np.sum(np.square(W1))+np.sum(np.square(W2)))  #代价函数添加L2参数正则项
    return  loss #返回代价函数值

def back_prop(X,y,model,z1,a1,z2,output,reg_lambda):
    """
    Function: 通过反向传播计算各权重与偏置的导数
    Parameters: X - 样本数据矩阵
                y - 样本数据真实类别标签矩阵
                model - 本次迭代的神经网络模型
                z1 - 本次迭代的隐藏层输入值
                a1 - 本次迭代的隐藏层输出值
                z2 - 本次迭代的输出层输入值
                output - 本次迭代的输出层输出值
                reg_lambda - 正则化参数
    Return: dW1 - 输入层到隐藏层权重的导数
            db1 - 隐藏层偏置的导数
            dW2 - 隐藏层到输出层权重的导数
            db2 - 输出层偏置的导数
    """
    num_examples = X.shape[0] #样本数据个数
    delta3_output = output-y #代价函数对softmax输出层输入z2的导数 ,C/zi=softmax(zi)-yiLabel
    dW2=np.zeros(model['W2'].shape) #初始化权重W2导数
    db2=np.zeros(model['b2'].shape) #初始化偏置b2导数
    dW1=np.zeros(model['W1'].shape) #初始化权重W1导数
    db1=np.zeros(model['b1'].shape) #初始化偏置b1导数
    for num in range(num_examples):
        delta3=delta3_output[num].reshape(1,-1) #对每一个样本，求取导数并累加
        dW2 +=np.dot(a1[num].reshape(-1,1),delta3)  #计算对权重W2的导数,可以理解为W2(i,j)通过与a1[i]影响到z2[j]
        db2 +=delta3 #计算对偏置b2的导数
        delta2=np.dot(delta3,model['W2'].T)*ReLU_derivative(z1[num].reshape(1,-1)) #计算对隐藏层输入z1的导数，采用整流线性单元求导
        dW1 +=np.dot(X[num].reshape(-1,1),delta2) #计算对权重W1的导数
        db1 +=delta2   #计算对偏置b1的导数
    dW2=dW2/num_examples + reg_lambda*model['W2'] #取平均，加上正则项
    dW1=dW1/num_examples + reg_lambda*model['W1'] #取平均，加上正则项
    db2=db2/num_examples #取平均
    db1=db1/num_examples #取平均
    return dW2,db2,dW1,db1 #返回各权重与偏置的导数

def train(model,X,y,num_iter=10000,reg_lambda=0.1,learning_rate=0.1):
    """
    Function:基于样本数据训练神经网络模型
    Parameters: model - 初始化的神经网络
                X - 样本数据矩阵
                y - 样本数据真实类别标签矩阵
                num_iter - 迭代次数阈值
                reg_lambda - 正则化超参数
                learning_rate - 学习速率
    Return: model -训练好的模型
            losses - 损失序列
    """
    done=False #初始化训练是否完成的标志位
    previous_loss = float('inf') #初始化上个迭代损失
    i=0 #初始化当前迭代次数
    losses=[] #初始化迭代损失序列
    while(i<num_iter and done==False): #当迭代次数小于阈值且训练没有达到目标，继续迭代
        if i%100 ==0: #当迭代次数为100的整数倍时，计算损失，并与上一个迭代损失比较
            loss=calculate_loss(model,X,y,reg_lambda) #计算本次迭代的损失
            losses.append(loss) #将损失添加到损失序列
            if(previous_loss - loss) < 0.00001: #如果相对上一个损失的减小变化小于某个范围，停止迭代/previous_loss
                done = True
            previous_loss = loss #更新上个迭代损失
        z1,a1,z2,output=feed_forward(model,X)  #计算本次迭代各层的输入输出
        dW2,db2,dW1,db1 =back_prop(X,y,model,z1,a1,z2,output,reg_lambda) #计算本次迭代各权重与偏置的导数
        #更新权重与偏置
        model['W2']-=learning_rate*dW2
        model['b2']-=learning_rate*db2
        model['W1']-=learning_rate*dW1
        model['b1']-=learning_rate*db1
        i+=1
    return model,losses #返回训练好的模型、迭代损失序列


if __name__=='__main__':
    X,y=datasets.make_moons(256,noise=0.10) #利用datasets生成二分类数据
    num_examples = len(X) #样本数据个数
    nn_output_dim = 2 #输出层维数，即实现二分类
    labelMat=np.zeros((num_examples,nn_output_dim)) #零值初始化样本真实类别标签
    for num in range(num_examples):
        labelMat[num,y[num]]=1  #根据样本数据真实类别标签将相应类别标志为1
    learning_rate = 0.01 #学习速率
    reg_lambda = 0.01 #正则化超参数
    model = built_model(X,20,nn_output_dim) #随机初始化模型
    model,losses = train(model,X,labelMat,10000,reg_lambda,learning_rate)#训练模型
    z1,a1,z2,output = feed_forward(model,X) #利用训练好的模型预测原有样本数据
    pred =np.argmax(output,axis=1) #得到样本数据预测的类别标签
    error_rate=0 #初始化预测错误率 
    usedMarker=['o','v'] #定义两种散点类型
    usedColor=['g','r'] #定义两种散点颜色
    plt.title('moons_data_classify')
    for num in range(num_examples): #统计预测错误个数
        plt.scatter(X[num,0],X[num,1],marker=usedMarker[y[num]],c=usedColor[pred[num]]) #绘制散点，真实类别用标记形状区分，预测类别用颜色区分
        if(y[num] != pred[num]):
            error_rate+=1
    error_rate=error_rate/num_examples #错误率
    plt.show() #绘图显示

    

    
    