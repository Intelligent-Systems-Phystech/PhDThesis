# coding: utf-8
import numpy as np
import sys
sys.path.append('.')

from models.feedforward import build_feedforward
from generic.optimizer import gd_optimizer
from generic.regularizers import gaus_prior
from functools import partial 
from tc.simple import  simple_tc
from hyperoptimizers.random_search import random_optimize
import theano
#get_ipython().magic(u'matplotlib inline')
import matplotlib.pylab as plt

import matplotlib.pylab as plt

X = np.random.randn(100, 2)
Y = X[:,0]
lr = theano.shared(1.0)
alpha = theano.shared(np.array([1.0])) 
print id(alpha)

optimizer = partial(gd_optimizer, learning_rate=lr)


model_build = partial(build_feedforward, structure = [2,1], nonlinearity=lambda x:x, alphas=alpha, dataset_size=100)

random_optimize(partial(simple_tc, batch_size=10), model_build, optimizer, 10, 100, X, Y,  [alpha, lr] ,
 [[np.array([1.0]), np.array([0.1]), np.array([0.001]), np.array([0.001])], [10**(-5)]])
