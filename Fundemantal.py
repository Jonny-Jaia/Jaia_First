import  numpy as np

def identity_func(x):
    return x
    pass

def step_func(x):
    return np.array(x>0,dtype=np.int)
    pass

def sigmoid(x):
    return 1/(1+np.exp(-x))
    pass

def relu(x):
    return np.maximum(0,x)
    pass

def sigmoid_grad(x):
    return (1.0-sigmoid(x))*sigmoid(x)
    pass

def relu_grad(x):
    grad=np.zeros(x)
    grad[x>=0]=1
    return grad
    pass

def softmax(x):
    if x.ndim==2:
        x=x.T
        x=x-np.max(x,axis=0)##列项
        y=np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T
        pass

    x=x-np.max(x)##溢出的解决办法
    return np.exp(x)/np.sum(np.exp(x))
    pass

def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)
    pass

def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)

    if t.size==y.size:
        t=t.argmax(axis=1)

    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size),t]+1e-7))/batch_size
    pass

def softmax_loss(x,t):
    y=softmax(x)
    return cross_entropy_error(y,t)
    pass
