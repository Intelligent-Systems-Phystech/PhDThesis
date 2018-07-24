import numpy as np
import sys
#sys.path.append('../../')
#sys.path.append('../../pyfos/')
from models.feedforward import build_feedforward
from models.var_feedforward import build_var_feedforward
from generic.optimizer import gd_optimizer
from generic.regularizers import gaus_prior
from functools import partial 
from tc.simple import  simple_tc
from tc.cv import  cv_tc
from hyperoptimizers.random_search import random_optimize
from hyperoptimizers.greed_optimize import greed_optimize
import theano

import matplotlib.pylab as plt
import random

matrix = np.load('../../data/matrix.npy')
X, Y = np.load('../../data/linearx.npy'), np.load('../../data/lineary.npy')
X_train = X[:100]
Y_train = Y[:100]
X_test = X[100:]
Y_test = Y[100:]
lr = theano.shared(0.005)
alphas = theano.shared(np.array([3.45387764,  0.      ]))

optimizer = partial(gd_optimizer, learning_rate=lr)
model_build = partial(build_var_feedforward,  structure = [2,1], nonlinearity=lambda x:x, log_alphas=alphas, bias=False, param_pool_size=100)






training_procedure = simple_tc( model_build, optimizer, X_train, Y_train, batch_size=100 )
    
for i in xrange(100):
    training_procedure.do_train()
    print training_procedure.models[0].params.eval()
