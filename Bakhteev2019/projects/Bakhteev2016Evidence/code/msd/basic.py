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
import cPickle as pickle
load_path = sys.argv[1]
save_path = sys.argv[2]

if not os.path.exists(save_path):
	os.mkdir(save_path)



lrs = [5*10**(-9), 2.5*10**(-9), 10**(-9), 5*10**(-10), 2.5*10**(-10), 10**(-10)]
alphas = [ 10.0, 10.0**(3), 10.0**(5), 10.0**(7), 10.0**(9) ]

train_x, train_y = np.load(os.path.join(load_path, 'train_x.npy')).astype(np.float32),np.load(os.path.join(load_path, 'train_y.npy')).astype(np.float32)

valid_part = 0.1
batch_size = 50
hidden_num = 100
patience = 3
start_num = 10
max_iter = 500000
print 'max iter is', max_iter

for lr in lrs:

	to_save = {'dataset_path': load_path, 'hidden_num':hidden_num, 'batch_size':batch_size,  'lr':lr, 'params':[], 'validation':[]}
	
	print 'lr', lr
	params = []	
	costs = []
	updates = []
	full_X = theano.shared(train_x)
	full_Y = theano.shared(train_y)
	indices = T.ivector()
	X = full_X[indices]
	Y = full_Y[indices]

	#http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
	for start in range(start_num):		
		prior_std1 = np.sqrt(2.0)/np.sqrt(train_x.shape[1]+hidden_num)
		prior_std2 = np.sqrt(2.0)/np.sqrt(hidden_num+1)
		multiplier =  [prior_std1]*(train_x.shape[1] * hidden_num ) + [prior_std2]*hidden_num*1+[prior_std1]*hidden_num+[prior_std2]*1
		param = theano.shared((np.random.randn(train_x.shape[1]*hidden_num+hidden_num*1+1+hidden_num).astype(np.float32) * multiplier).astype(np.float32))

		params.append(param)
		W1 = param[:train_x.shape[1]*hidden_num].reshape((train_x.shape[1],hidden_num))
		b1 = param[-hidden_num-1:-1]
		W2 = param[train_x.shape[1]*hidden_num:train_x.shape[1]*hidden_num+hidden_num*1].reshape((hidden_num, 1))
		b2 = param[-1:]
		hidden =T.log(1+T.exp(T.dot(X, W1) + b1))#
		output = T.dot(hidden, W2) + b2
		cost = T.sum((output.T - Y)**2)*train_x.shape[0]/X.shape[0]
		costs.append(cost)
		grad = T.grad(cost, wrt= param)
		updates.append((param, param - lr * grad))
	train = theano.function([indices], T.mean(costs), updates= updates)
	monitor = theano.function([indices], -T.mean(costs))
	best_cost = -999999999999
	import time
	cur_patience = patience
	time_start = time.time()
	for i in xrange(0, max_iter):
		sample = random.sample(range(train_x.shape[0]), batch_size)
		time_s = time.time()
		batch_result = train(sample)

		if i%1000==0 or i==max_iter-1:
			to_save['params'].append([p.eval() for p in params])

			sample = range(train_x.shape[0])
			result = monitor(sample)/train_x.shape[0]
			print 'tf', time.time() - time_start
			time_start = time.time()
			to_save['validation'].append(result)
			with open(os.path.join(save_path, str(lr)+'.pckl'), 'wb') as out:
				pickle.dump(to_save, out)

			print 'validation', i, result
			if result > best_cost:				
				print '*'
				best_cost = result
				cur_patience = patience
			else:
				cur_patience-=1
			if cur_patience==0:
				print 'patience exit'
				break

	print 'result',lr, best_cost
	import time		
	time.sleep(60)

		
	


