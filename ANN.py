import os, sys
import math as m
import numpy as np
import copy
import time as t
import random as r

###     Values     ###
lay_C = 0

###     Lists     ###
nodes = []
synapses = []
link = []
syn_deltas = []
FD = []
time_w = []

def sig(x, d=False):
    if d == True:
        return np.exp(-x)*(1-np.exp(-x))**(-2)
    return 1.0/(1.0-np.exp(-x))

def tanh(x, d=False):
    if d == True:
        return (4*np.exp(2*x))/((np.exp(2*x)-1)**2)
    return (np.exp(2*x)-1)/(np.exp(2*x)+1)

def Init(id, hd, od):
    fd = [id] + [hd[0] for i in xrange(hd[1])] + [od]
    FD = fd
    lay_C = len(fd)-1
    #np.random.seed(0)
    for i in xrange(lay_C):
        synapses.append(2.0*np.random.random((fd[i], fd[i+1]))-1.0)
        link.append(np.random.randint(0,2,size=(fd[i],fd[i+1])))
        s = link[i].shape
        try:
            link[i][0,0:] = 1
            link[i][(s[0]-1):,0:] = 1
        except IndexError:
            pass
        #print link[i]
    for i in synapses:
        l = []
        for j in i:
            l.append(np.sum(j**2))
        time_w.append(sum([k**2 for k in l]))

def activate(il):
    global nodes
    il = np.array(il)
    try:
        nodes.append(tanh(np.dot(il.T, (synapses[0]*link[0]))))#*link[0]
        drop_perc = 0.5
        for i in xrange(lay_C-1):
            nodes[i] *= np.random.binomial([np.ones((FD[i], FD[i+1]))], 1-drop_perc)[0] * (1.0/(1-drop_perc))# dropout
            nodes.append(tanh(np.dot(nodes[i], (synapses[i+1]*link[i+1]))))#*link[i+1]
    except ValueError:
        pass

def LMSE(Y, y, n):
    return np.sum((Y-y)**2)/n

def Train(il, y, n, alpha=0.01):       #Backpropagation wi Gradient descent and the adaGrad optimization
    global nodes
    nodes = []
    activate(il)
    syn_deltas.append((nodes[-1]-y)*tanh(nodes[-1],d=True))
    for i in xrange(lay_C-1):
        syn_deltas.append(syn_deltas[-(i+1)].dot(synapses[-i].T)*tanh(nodes[-(i+1)], d=True))
    for i in xrange(lay_C-2):
        synapses[-(i+1)] -= (((alpha)/m.sqrt(time_w[-(i+2)] + 1)) * nodes[-(i+2)].T.dot(syn_deltas[i+1]))# adaGrad
    synapses[0] -= (((alpha)/m.sqrt(time_w[0] + 1)) * il.dot(syn_deltas[-1])) #adaGrad
    activate(il)
    lmse = LMSE(nodes[-1], y, n)
    for i in xrange(lay_C-1):
        l = []
        for j in synapses[i]:
            l.append(np.sum(j**2))
        time_w[i] += (sum([k**2 for k in l]))
    return lmse


def Batch_Learning(training_set, n):
    err = []
    for s in training_set:
        err.append(Train(np.array(s[0]), np.array(s[1]), n))
    return sum([i for i in err])
    
#########################################################
#########################################################
#               The Actual Running Process              #
#########################################################
#########################################################

inDim, hDim, oDim = [4,(4,1),1]
Init(inDim, hDim, oDim)
In = [np.array([[0.1],[0.1],[1],[0.1]]),
      np.array([[0.1],[0.1],[0.1],[0.1]]),
      np.array([[0.1],[0.1],[1],[1]])]
Out = [np.array([[0.1,0.1,1,1]]),
       np.array([[0.1, 0.1, 0.1,1]]),
       np.array([[0.1, 1, 0.1,0.1]])]
lerr = 0
i = 1
for i in xrange(1):
    strt = t.clock()
    layer_vals = []
    activate(In[0])
    print nodes[-1]
    activate(In[1])
    print nodes[-1]
    training_set = np.array([In,Out])
    training_set = training_set.T
    #print training_set[0]
    while 1:
        err = Batch_Learning(training_set, i+1)
        if err < 0.0001: #This is the least mean square of the learnt data after running the training set
            end = t.clock()
            break
        #print err
        i += 1
    layer_vals = []
    activate(In[0])
    print In[0], nodes[-1]
    activate(In[1])
    print In[1], nodes[-1]
    print "Time spent training: %f seconds"%((end-strt))
print err
