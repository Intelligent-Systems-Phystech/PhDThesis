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
from framework.lang_estimator import LangevinEstimator

load_path = sys.argv[1]
save_path = sys.argv[2]

if not os.path.exists(save_path):
    os.mkdir(save_path)




lrs =  [5*10**(-3), 2.5*10**(-3), 10**(-3), 5*10**(-4), 2.5*10**(-4), 10**(-4), 5*10**(-5), 2.5*10**(-5), 10**(-5)]
alphas = [10.0,   10.0**(3), 10.0**(5), 10.0**(7), 10.0**(9) ]

train_x, train_y = np.load(os.path.join(load_path, 'train_x.npy')).astype(np.float32),np.load(os.path.join(load_path, 'train_y.npy')).astype(np.float32)
valid_part = 0.1
batch_size = train_x.shape[0]
hidden_num = 50
patience = 3
start_num = 10
max_iter = 25000
print 'max iter is', max_iter



def builder_sgd(param,X, Y):
    W1 = param[:train_x.shape[1]*hidden_num].reshape((train_x.shape[1],hidden_num))
    W2 = param[train_x.shape[1]*hidden_num:train_x.shape[1]*hidden_num+hidden_num*hidden_num].reshape((hidden_num, hidden_num))
    W3 = param[train_x.shape[1]*hidden_num+hidden_num*hidden_num:train_x.shape[1]*hidden_num+2*hidden_num**2].reshape((hidden_num,  hidden_num))
    W4 = param[train_x.shape[1]*hidden_num+2*hidden_num**2:train_x.shape[1]*hidden_num+2*hidden_num**2+hidden_num].reshape((hidden_num, 1))

    b1 = param[-hidden_num*3-1:-hidden_num*2-1]
    b2 = param[-hidden_num*2-1:-hidden_num-1]
    b3 = param[-hidden_num*1-1:-1]
    b4 = param[-1:]


    hidden1 =T.log(1+T.exp(T.dot(X, W1) + b1))#
    hidden2 =T.log(1+T.exp(T.dot(hidden1, W2) + b2))#
    hidden3 =T.log(1+T.exp(T.dot(hidden2, W3) + b3))#
    output = T.dot(hidden3, W4) + b4

    cost = T.sum((output.T - Y)**2)*train_x.shape[0]/X.shape[0]
       
    return cost




for lr in lrs:
    for alpha in alphas:
        alpha = np.float32(alpha)
        lr = np.float32(lr)
        to_save = {'dataset_path': load_path, 'hidden_num':hidden_num, 'batch_size':batch_size,  'lr':lr, 'alpha':alpha, 'params':[], 'validation':[]}
    
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
            
            
        prior_std1 = np.sqrt(2.0)/np.sqrt(train_x.shape[1]+hidden_num)#0-1
        prior_std2 = np.sqrt(2.0)/np.sqrt(hidden_num*2)#1-2
        prior_std3 = np.sqrt(2.0)/np.sqrt(hidden_num*2)#2-3
        prior_std4 = np.sqrt(2.0)/np.sqrt(hidden_num+1)
        
        multiplier =  [prior_std1]*(train_x.shape[1] * hidden_num ) + [prior_std2]*hidden_num**2+[prior_std3]*hidden_num**2+[prior_std4]*hidden_num+[prior_std1]*hidden_num+[prior_std2]*hidden_num + [prior_std3]*hidden_num + [prior_std4] * 1
        

        param_num = len(multiplier)
        estimator = LangevinEstimator(param_num, multiplier, X, Y, indices, lambda p,x,y:builder_sgd(p,x,y), lr=lr, prior_scale=[alpha], multistart=start_num)    
        best_cost = -999999999999
        cur_patience = patience
        import time
        time_s = time.time()
        for i in xrange(0, max_iter):
            estimator.lr.set_value((lr* (i+500)**(-0.51)).astype(np.float32))            
            sample = random.sample(range(train_x.shape[0]), batch_size)
            batch_result = estimator.update(sample)-estimator.prior()    
            if i%100==0 or i==max_iter-1:
                if i == max_iter-1:
                    to_save['params'].append([p.eval() for p in params])
                print 'current lr', estimator.lr.eval()                
                result,_ = estimator.evidence(range(train_x.shape[0]))
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
                    to_save['params'].append([p.eval() for p in estimator.params])
                    with open(os.path.join(save_path, str(lr)+'_'+str(alpha)+'.pckl'), 'wb') as out:
                        pickle.dump(to_save, out)
                    break

        print 'result',lr, alpha, best_cost
        import time        
        time.sleep(60)

        
    


