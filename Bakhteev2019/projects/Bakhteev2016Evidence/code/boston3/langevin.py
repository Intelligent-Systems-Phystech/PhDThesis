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


lrs = [ 10**(-5), 10**(-6), 10**(-7), 10**(-8), 10**(-9)]
alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

train_x, train_y = np.load(os.path.join(load_path, 'train_x.npy')).astype(np.float32),np.load(os.path.join(load_path, 'train_y.npy')).astype(np.float32)
batch_size = 25
hidden_num = 50
patience = 5
start_num = 10
max_iter = 300000

for lr in lrs:
	for alpha in alphas:
		to_save = {'dataset_path': load_path, 'hidden_num':hidden_num, 'batch_size':batch_size,  'lr':lr, 'params':[], 'validation':[]}
	
		print 'lr', lr
		print 'alpha', alpha
		params = []	
		costs = []
		updates = []
		X = T.matrix()
		Y = T.vector()
		for start in range(start_num):		
			prior_std1 = 2.0/np.sqrt(hidden_num+train_x.shape[1])
			prior_std2 = 2.0/np.sqrt(hidden_num+1)
			multiplier =  [prior_std1]*(train_x.shape[1] * hidden_num ) + [prior_std2]*hidden_num*1+[prior_std1]*hidden_num+[prior_std2]*1
			param = theano.shared(np.random.randn(train_x.shape[1]*hidden_num+hidden_num*1+1+hidden_num) * multiplier)
			params.append(param)
			W1 = param[:train_x.shape[1]*hidden_num].reshape((train_x.shape[1],hidden_num))
			b1 = param[-hidden_num-1:-1]
			W2 = param[train_x.shape[1]*hidden_num:train_x.shape[1]*hidden_num+hidden_num*1].reshape((hidden_num, 1))
			b2 = param[-1:]
			hidden = T.tanh(T.dot(X, W1)+b1)
			output = T.dot(hidden, W2) + b2
			cost = T.sum((output.T - Y)**2)*train_x.shape[0]/X.shape[0]
			costs.append(cost)
			grad = T.grad(cost, wrt= param)
			updates.append((param, param - lr * grad))
		train = theano.function([X, Y], T.mean(costs), updates= updates)
		monitor = theano.function([X, Y], -T.mean(costs))
		best_cost = -999999999999
		cur_patience = patience
		for i in xrange(0, max_iter):
			sample = random.sample(range(train_x.shape[0]), batch_size)
			batch_result = train(train_x[sample], train_y[sample])
            if i%100==0 or i==max_iter-1:
                if i == max_iter-1:
			        to_save['params'].append([p.eval() for p in params])
				
				result = monitor(train_x, train_y)/train_x.shape[0]
				to_save['validation'].append(result)
				with open(os.path.join(save_path, str(lr)+str(alpha)+'.pckl'), 'wb') as out:
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

		print 'result',lr, alpha, best_cost

		
	


