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




#lrs = [ 5*10**(-5),  2.5*10**(-5), 10**(-5), 5*10**(-6), 2.5*10**(-6), 10**(-6), 5*10**(-7), 2.5*10**(-7), 10**(-7) ]
lrs = [10**(-5), 2.5*10**(-5), 10**(-5), 5*10**(-6), 2.5*10**(-6), 10**(-6), 5*10**(-7), 2.5*10**(-7), 10**(-7) ]
#[ 5*10**(-5),  2.5*10**(-5), 10**(-5), 5*10**(-6), 2.5*10**(-6), 10**(-6), 5*10**(-7), 2.5*10**(-7), 10**(-7) ]

#lrs = [ 5*10**(-3), 2.5*10**(-3), 10**(-3), 5*10**(-4), 2.5*10**(-4), 10**(-4), 5*10**(-5), 2.5*10**(-5), 10**(-5) ]
alphas = [ 10.0,   10.0**(3), 10.0**(5), 10.0**(7), 10.0**(9) ]


train_x, train_y = np.load(os.path.join(load_path, 'train_x.npy')).astype(np.float32),np.load(os.path.join(load_path, 'train_y.npy')).astype(np.uint)




train_y[train_y == 7] = 0
train_y[train_y == 9] = 1

valid_part = 0.1
batch_size = 100
hidden_num = 50
patience = 3
start_num = 3#25
max_iter = 500000
print 'max iter is', max_iter

def builder_gaus(param, X_t, Y_t, neuron_num=hidden_num):
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
    
    output = T.nnet.sigmoid(T.dot(hidden3, W4) + b4).flatten()
    output = T.clip(output, 0.001, 0.999)
    cost_ =-np.dot(Y, T.log(output)) - np.dot((1-Y) , T.log(1-output))
    cost = T.sum(cost_)*train_x.shape[0]/X.shape[0]


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
            
        prior_std1 = np.sqrt(2.0)/np.sqrt(train_x.shape[1]+hidden_num)#0-1
        prior_std2 = np.sqrt(2.0)/np.sqrt(hidden_num*2)#1-2
        prior_std3 = np.sqrt(2.0)/np.sqrt(hidden_num*2)#2-3
        prior_std4 = np.sqrt(2.0)/np.sqrt(hidden_num+1)
        
        multiplier =  [prior_std1]*(train_x.shape[1] * hidden_num ) + [prior_std2]*hidden_num**2+[prior_std3]*hidden_num**2+[prior_std4]*hidden_num+[prior_std1]*hidden_num+[prior_std2]*hidden_num + [prior_std3]*hidden_num + [prior_std4] * 1
        

        param_num = len(multiplier)
        estimator = GausEstimator(param_num, multiplier, X, Y, indices, lambda p,x,y:builder_gaus(p,x,y), lr=lr, prior_scale=np.array([alpha]*param_num), multistart=start_num, batch_size=100)            
        best_cost = -999999999999
        cur_patience = patience
        import time
        time_s = time.time()
        for i in xrange(0, max_iter):
            sample = random.sample(range(train_x.shape[0]), batch_size)
            batch_result = estimator.update(sample)
            if i%100==0 or i==max_iter-1:
                if i == max_iter-1:
                    to_save['params'].append([p.eval() for p in params])
                
                time_s = time.time()
            
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
                    to_save['params'].append([p.eval() for p in estimator.params_all])
                    print to_save
            
                    with open(os.path.join(save_path,str(lr)+'_'+str(alpha)+'.pckl'), 'wb') as out:
                        pickle.dump(to_save, out)
                    break

        print 'result',lr, alpha, best_cost
        import time        
        time.sleep(60)
        
    


