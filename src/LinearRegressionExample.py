
# coding: utf-8

# In[1]:

import numpy as np
import math
import matplotlib.pyplot as plt


# In[2]:

## 배치 버전


# In[3]:

def f(x):
    w = 2.0
    return w * x

def perturb(y):
    mu, sigma = y, 2.0
    return np.random.normal(mu, sigma)

def makeData():
    x_list = list(np.random.uniform(0, 100, 100))
    x_list.sort()
    data = map(lambda x: (x, perturb(f(x))), x_list)
    return data

def gradE(data, w):
    return sum(map(lambda (x, y): float(x * (w * x - y)), data))

def updateW(data, w, eta):
    return w - eta * gradE(data, w)

data = makeData()
def searchW():
    ## data = makeData()
    w = 0.0
    eta = 1.0e-6

    iter = 20
    for i in range(iter):
        w = updateW(data, w, eta)
    return w
    
print searchW()


# In[4]:

x_list = np.array(map(lambda x: x[0], data))
y_list = np.array(map(lambda x: x[1], data))
plt.plot(x_list, y_list)
plt.show()


# In[ ]:




# In[5]:

## 온라인 버전


# In[6]:

def f(x):
    w = 2.0
    return w * x

def perturb(y):
    mu, sigma = y, 1.0
    return np.random.normal(mu, sigma)

def makeData():
    x_list = list(np.random.uniform(0, 100, 100))
    x_list.sort()
    data = map(lambda x: (x, perturb(f(x))), x_list)
    return data

def gradE(example, w):
    x, y = example
    return float(x * (w * x - y))

def updateW(example, w, eta):
    return w - eta * gradE(example, w)

def searchW():
    data = makeData()
    w = 0.0
    eta = 1.0e-4

    for example in data:
        w = updateW(example, w, eta)
    return w
    
print searchW()


# In[ ]:














# In[7]:

def f(x):
    w1 = 2.0
    w2 = 10.0
    return w1 * x + w2

def perturb(y):
    mu, sigma = y, 1.0
    return np.random.normal(mu, sigma)

def makeData():
    x_list = list(np.random.uniform(0, 100, 100))
    x_list.sort()
    data = map(lambda x: (x, perturb(f(x))), x_list)
    return data

def gradE(data, w):
    w1, w2 = w
    return (sum(map(lambda (x, y): float(x * (w1 * x + w2 - y)), data)), sum(map(lambda (x, y): float(w1 * x + w2 - y), data)))

def updateW(data, w, eta):
    w1, w2 = w
    eta1, eta2 = eta
    gradE_w1, gradE_w2 = gradE(data, w)

    return (w1-eta1*gradE_w1, w2-eta2*gradE_w2)

def searchW():
    data = makeData()
    w = (0.0, 0.0)
    eta = (1.0e-6, 1.0e-4)

    iter = 10000
    for i in range(iter):
        w = updateW(data, w, eta)
        ## print w
    return w
    
print searchW()


# In[ ]:




# In[8]:

def f(x):
    w1 = 2.0
    w2 = 10.0
    return w1 * x + w2

def perturb(y):
    mu, sigma = y, 1.0
    return np.random.normal(mu, sigma)

def makeData():
    x_list = list(np.random.uniform(0, 100, 100))
    x_list.sort()
    data = map(lambda x: (x, perturb(f(x))), x_list)
    return data

def gradE(example, w):
    x, y = example
    w1, w2 = w
    return (float(x * (w1 * x + w2 - y)), float((w1 * x + w2 - y)))

def updateW(example, w, eta):
    w1, w2 = w
    eta1, eta2 = eta
    gradE_w1, gradE_w2 = gradE(example, w)

    return (w1-eta1*gradE_w1, w2-eta2*gradE_w2)

def searchW():
    data = makeData()
    w = (0.0, 0.0)
    eta = (1.0e-5, 1.0e-3)

    for i in range(500):
        for example in data:
            w = updateW(example, w, eta)
            ## print w
    return w
    
print searchW()


# In[ ]:



