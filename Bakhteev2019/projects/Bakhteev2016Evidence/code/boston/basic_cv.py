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


lrs = [ 5*10**(-5), 2.5*10**(-5), 10**(-5), 5*10**(-6), 2.5*10**(-6), 10**(-6)]

train_x, train_y = np.load(os.path.join(load_path, 'train_x.npy')).astype(np.float32),np.load(os.path.join(load_path, 'train_y.npy')).astype(np.float32)
valid_part = 0.1
batch_size = int(train_x.shape[0]*(1-valid_part))
hidden_num = 50
patience = 5
start_num = 10
max_iter = 25000
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

	train_functions = []
	monitor_functions = []
	validation_indexes = []
	train_indexes = []
	for start in range(start_num):		
		validation_indexes.append(random.sample(range(train_x.shape[0]), int(valid_part*train_x.shape[0])))
		train_indexes.append([i for i in range(train_x.shape[0]) if i not in validation_indexes[-1]])
		prior_std1 = np.sqrt(2.0)/np.sqrt(train_x.shape[1])
		prior_std2 = np.sqrt(2.0)/np.sqrt(hidden_num)
		multiplier =  [prior_std1]*(train_x.shape[1] * hidden_num ) + [prior_std2]*hidden_num*1+[prior_std1]*hidden_num+[prior_std2]*1
		param = theano.shared((np.random.randn(train_x.shape[1]*hidden_num+hidden_num*1+1+hidden_num) * multiplier).astype(np.float32))
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
		updates = ([(param, param - lr * grad)])
		train = theano.function([indices], cost, updates = updates)
		train_functions.append(train)
		monitor = theano.function([indices], -cost)
		monitor_functions.append(monitor)
	best_cost = -999999999999
	cur_patience = patience
	import time
	time_s = time.time()
	for i in xrange(0, max_iter):
		batch_result = []
		for start in range(start_num):		
			sample = random.sample(train_indexes[start], batch_size)
			batch_result.append(train_functions[start](sample))
		
		if i%100==0 or i==max_iter-1:
			print time.time() - time_s
			time_s = time.time()			
			to_save['params'].append([p.eval() for p in params])

			result = []
			for start in range(start_num):	
				sample = validation_indexes[start]
				result.append(monitor_functions[start](sample))
			
			result = np.mean(result)/train_x.shape[0]
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

		
	


