import numpy as np

class SGD:
    ##随机梯度下降
    def __init__(self,lr=0.01):
        self.lr=lr
        pass
    def update(self,params,grads):
        for key in params.keys():
            params[key]-=self.lr*grads[key]
            pass
        pass

class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr=lr
        self.momentum=momentum
        self.v=None
        pass
    def update(self,params,grads):
        if self.v==None:
            self.v={}
            for key,val in params.items():
                self.v[key]=np.zeros_like(val)

        for key in params.key():
            self.v[key]=self.momentum*self.v[key]-self.lr*grads
            params[key]+self.v[key]
        pass

class AdaGrad:
    def __index__(self,lr=0.01):
        self.lr=lr
        self.h=None
        pass

    def update(self,params,grads):
        if self.h==None:
            self.h={}
            for key,val in params.items():
                self.h[key]=np.zeros_like(val)

        for key in params.key():
            self.h[key]+=grads[key]*grads[key]
            params[key]-=self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7)
        pass

class Adam:
    def __index__(self, lr=0.001,beta1=0.9,beta2=0.09):
        self.lr = lr
        self.beta1 =beta1
        self.beta2 = beta2
        self.iter=0
        self.m=None
        self.v=None
        pass

    def update(self, params, grads):
        if self.m == None:
            self.m,self.v = {},{}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter+=1
        lr_t=self.lr*np.sqrt(1-self.beta2**self.iter)/(1.0-self.beta1**self.iter)
        for key in params.key():
            self.m[key] += (1-self.beta1)*(grads[key] -self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
        pass
