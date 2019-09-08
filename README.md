# DL_MLP_CNN_RNN_LSTM_Numpy
Basic DL algorithms implemented with Numpy ,inlcuding MLP, CNN, simple RNN, basic LSTM.

本程序基于Numpy模块实现深度学习中的一些基本算法，包括多层感知机、卷积神经网路、循环神经网络、长短期记忆网络。

1、 npdl模块——微型的深度学习库
Activation.py——激活函数类，ReLU、Softmax、Tanh、Sigmoid
Cost.py——代价函数，softmax函数交叉熵损失函数
Initial.py——初始化类，Zero、Uniform、He_Uniform、Glorot_Uniform、Orthogonal
Layers.py——神经网络层类，Convolution、MeanPooling、Flatten、Softmax、SimpleRNN、LSTM、Embedding
Optimizers.py——优化器类，SGD随机梯度下降法
Model.py——神经网络模型类，包括各层神经网络组合连接，模型训练，模型预测等，初始化该类实例便可构建模型

2、神经网络实现
CNN_numpy_MNIST.py——通过CNN实现minist数据集的分类
LSTM_numpy_sentence.py——基于LSTM实现简单的语句词性分析
RNN_numpy_character.py——基于RNN实现简单的文本分类
threeLayer_MLP.py——简单的多层感知机实现
