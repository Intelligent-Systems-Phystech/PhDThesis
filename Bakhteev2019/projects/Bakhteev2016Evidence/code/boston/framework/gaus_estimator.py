from abstract_estimator import AbstractEstimator
import theano 
import theano.tensor as T
import numpy as np
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
np.random.seed(42)	
class GausEstimator(AbstractEstimator):
	def __init__(self,  param_length, param_scale,  X, Y, ind, builder, **kwargs):
		self.multistart = kwargs.get('multistart',1)
		self.lr =  theano.shared(kwargs.get('lr',0.001))
		self.prior_scale = np.array(kwargs.get('prior_scale',param_scale)) #non square!
		self.param_length = param_length
		#self.mean =  np.array(kwargs.get('mean',np.zeros((self.param_length, ))))
		self.batch_size  = int(kwargs.get('batch_size', 1))
		srng = RandomStreams()
		self.params_all = []
		self.updates = []
		self.cost = []
		for _ in range(self.multistart):
			_params = np.zeros(param_length*2)
			_params[param_length:] = np.log(np.array(param_scale))#non square!
			_params = theano.shared(_params.astype(np.float32))
			mean = _params[:param_length]
			param_log_std = _params[param_length:]
			params = srng.normal((X.shape[0],param_length))*T.exp(param_log_std)   + mean
			model = builder(params, X, Y)
			KLD = 0.5*  (T.sum(T.exp(2*param_log_std)/self.prior_scale**2) - param_length  - T.sum(T.log(T.exp(2*param_log_std))) + np.sum(np.log((np.array(self.prior_scale)**2)))  + T.dot(mean, mean)/T.dot(self.prior_scale, self.prior_scale)) #N_1 - prior, N_0 - out parameters
			cost = model + KLD
			grad = T.grad(cost, _params)
			self.params_all.append(_params)
			self.updates.append((_params, _params - self.lr*grad))
			self.cost.append(cost)
		#_update = theano.function([ind], grad)
		self.update = theano.function([ind], T.mean(self.cost), updates =self.updates)
		#self.likelihood = theano.function([ind], -T.mean(self.cost)self.model	)
		self._evidence = theano.function([ind], [-T.mean(self.cost)])
		
	


	def evidence(self, ind):
		#every _evidence result is an evidence estimation / batch_size
		#so, there we make mean of evidence/batch_size, which is ok to use as an estimate within validation
		result = 0.0
		cnt = 0
		for q in xrange(0, len(ind), self.batch_size):
			cnt+=1
			if q%1000==0:
				print q
			sample = range(q,min(len(ind), q+self.batch_size))
			result = result + self._evidence(sample)[0]
		return result/cnt
		


		
