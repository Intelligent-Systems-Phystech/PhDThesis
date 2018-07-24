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
from framework.gaus_estimator2 import GausEstimator

load_path = sys.argv[1]
save_path = sys.argv[2]

if not os.path.exists(save_path):
	os.mkdir(save_path)




lrs = [2.5*10**(-9), 10**(-9)]#, 5*10**(-10), 2.5*10**(-10), 10**(-10)]
alphas = [ 10.0, 10.0**(3), 10.0**(5), 10.0**(7), 10.0**(9) ]

train_x, train_y = np.load(os.path.join(load_path, 'train_x.npy')).astype(np.float32),np.load(os.path.join(load_path, 'train_y.npy')).astype(np.float32)

valid_part = 0.1
batch_size = 50
hidden_num = 100
patience = 3
start_num = 3
max_iter = 500000
print 'max iter is', max_iter

def builder_gaus(params, X_t, Y_t, neuron_num=hidden_num):
   	 class_num = 1
         W1 = params[: train_x.shape[1] * neuron_num].reshape((train_x.shape[1], neuron_num))
         W2 = params[train_x.shape[1] * neuron_num:train_x.shape[1] * neuron_num+neuron_num*class_num].reshape((( neuron_num, class_num)))
         b1 = params[-neuron_num-class_num:-class_num].reshape(((neuron_num,)))
         b2 = params[-class_num:].reshape((( class_num,)))
         first_layer = T.dot(X_t,W1)+b1	

         output = T.dot(T.log(1+T.exp(first_layer)),W2)+b2
         cost = T.sum((output.T - Y)**2)*train_x.shape[0]/X.shape[0]
       
         return cost





for lr in lrs:
	for alpha in alphas:
		to_save = {'dataset_path': load_path, 'hidden_num':hidden_num, 'batch_size':batch_size,  'lr':lr, 'alpha':alpha, 'params':[], 'validation':[]}
		lr = np.float32(lr)
		print 'lr', lr
		print 'alpha', alpha
		params = []	
		costs = []
		updates = []
		full_X = theano.shared(train_x)
		full_Y = theano.shared(train_y)
		indices = T.ivector()
		X = full_X[indices]
		Y = full_Y[indices]
			
		prior_std1 = np.sqrt(2.0)/np.sqrt(train_x.shape[1])
		prior_std2 = np.sqrt(2.0)/np.sqrt(hidden_num)
		multiplier =  [prior_std1]*(train_x.shape[1] * hidden_num ) + [prior_std2]*hidden_num*1+[prior_std1]*hidden_num+[prior_std2]*1
		param_num = len(multiplier)
		estimator = GausEstimator(param_num, multiplier, X, Y, indices, lambda p,x,y:builder_gaus(p,x,y), lr=lr, prior_scale=np.array([alpha]*param_num), multistart=start_num)			
		estimator.batch_size = 500#for evidence estimation, not real batch size
		best_cost = -999999999999
		cur_patience = patience
		import time
		time_s = time.time()
		for i in xrange(0, max_iter):
			sample = random.sample(range(train_x.shape[0]), batch_size)
			batch_result = estimator.update(sample)
			if i%1000==0 or i==max_iter-1:
				print time.time() - time_s
				time_s = time.time()
				to_save['params'].append([p.eval() for p in estimator.params_all])
				result = estimator.evidence(range(train_x.shape[0]))
				to_save['validation'].append(result)
				with open(os.path.join(save_path, str(lr)+'_'+str(alpha)+'.pckl'), 'wb') as out:
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
		import time		
		time.sleep(60)
		
	


