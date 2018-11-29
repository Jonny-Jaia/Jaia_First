# coding: utf-8
import numpy as np
from PublicFunction.Fundemantal import *

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

##超参数优化
class Dropout:


    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class BatchNormalization:

    def __init__(self,gamma,beta,momentum=0.9,running_mean=None,running_var=None):
        """
        :param gamma:
        :param beta:
        :param momentum:
        :param runninh_mean:
        :param running_var:
        """
        self.gamma=gamma
        self.beta=beta
        self.momentum=momentum
        self.input_shape=None
        #Conv层的情况下为4维，全连接层的情况下为2维

        #测试时用的平方值和方差值
        self.running_mean=running_mean
        self.running_var=running_var

        #反向传播中用到的变量
        self.batch_size=None
        self.xc=None
        self.std=None
        self.dgamma=None
        self.dbeta=None

    def forward(self,x,train_flag=True):
        self.input_shape=x.shape
        if x.ndim!=2:
            N,C,H,W=x.shape
            x=x.reshape(N,-1)

        out=self.__forward(x,train_flag)

        return out.reshape(*self.input_shape)

    def __forward(self,x,train_flag):
        if self.running_mean is None:
            N,D=x.shape
            self.running_mean=np.zeros(D)
            self.running_var=np.zeros(D)

        if train_flag:
            mu=x.mean(axis=0)
            






