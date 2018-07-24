"""
Args:
<path to data>, <path to results>
"""
import numpy as np
import theano
import theano.tensor as T
import logging
import random
import os
import sys
import time
import cPickle as pickle



lrs = [10**(-7),  10**(-8), 10**(-9), 10**(-10), 10**(-11)]

train_x, train_y = np.random.randn(50000, 15).astype(np.float32), np.random.randn(50000).astype(np.float32)

hidden_num = 25
lr = 10**(-7)
start_num = 30
params = []	
costs = []
updates = []
full_X = theano.shared(train_x)
full_Y = theano.shared(train_y)
indices = T.ivector()
X = full_X[indices]
Y = full_Y[indices]
max_iter = 10**7
#http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
for start in range(start_num):		
	prior_std1 = np.sqrt(2.0)/np.sqrt(train_x.shape[1])
	prior_std2 = np.sqrt(2.0)/np.sqrt(hidden_num)
	multiplier =  [prior_std1]*(train_x.shape[1] * hidden_num ) + [prior_std2]*hidden_num*1+[prior_std1]*hidden_num+[prior_std2]*1
	param = theano.shared((np.random.randn(train_x.shape[1]*hidden_num+hidden_num*1+1+hidden_num).astype(np.float32) * multiplier).astype(np.float32))

	params.append(param)
	W1 = param[:train_x.shape[1]*hidden_num].reshape((train_x.shape[1],hidden_num))
	b1 = param[-hidden_num-1:-1]
	W2 = param[train_x.shape[1]*hidden_num:train_x.shape[1]*hidden_num+hidden_num*1].reshape((hidden_num, 1))
	b2 = param[-1:]
	hidden = T.nnet.relu(T.dot(X, W1)+b1)
	output = T.dot(hidden, W2) + b2
	cost = T.sum((output.T - Y)**2)*train_x.shape[0]/X.shape[0]
	costs.append(cost)
	grad = T.grad(cost, wrt= param)
	updates.append((param, param - lr * grad))
train = theano.function([indices], T.mean(costs), updates= updates)
monitor = theano.function([indices], -T.mean(costs))
for i in xrange(0, max_iter):
	sample =  range(25)
	#time_s = time.time()
	#batch_result = train(sample)
	#print 'train',time_s  - time.time()
	time_s = time.time()
	batch_result = monitor(sample)
	print 'monitor', time_s  - time.time()
	
