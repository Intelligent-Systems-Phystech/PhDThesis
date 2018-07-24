from abstract_estimator import AbstractEstimator
import theano 
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
np.random.seed(42)	
class LangevinEstimator(AbstractEstimator):
	def __init__(self,  param_length, param_scale,  X, Y, builder, **kwargs):
		self.multistart = kwargs.get('multistart',1)
		self.lr =  theano.shared(kwargs.get('lr',0.001))
		self.prior_scale = kwargs.get('prior_scale',param_scale)
		self.param_length = param_length
		models = []
		updates = []
		srng = RandomStreams()
		self.rvn = srng.normal((param_length,))
		self.entropy_change = []
		priors = []
		self.params =[]
		for i in xrange(0, self.multistart):
			params = theano.shared(np.random.randn(param_length)*np.array(param_scale))
			neg_prior = 0.5 * param_length * np.log(2*np.pi) + np.sum(np.log(self.prior_scale)) + 0.5 * T.dot(params/self.prior_scale, params/self.prior_scale)
			model = builder(params, X, Y)
			cost = model + neg_prior
			grad = T.grad(cost, params)
			def jvp(vector):
				hvp = T.grad(T.dot(grad, vector), params)
				jvp = vector - self.lr * hvp 
				return jvp 
			
			r = T.vector()
			r1 = jvp(r)
			e_change = theano.function([r,X,Y], r1)
			self.entropy_change.append(e_change)
			self.params.append(params)
			updates.append((params, params - self.lr *  grad +self.rvn*T.sqrt(2*self.lr) ))
			models.append(model)
			
			priors.append(-neg_prior)

		self._update = theano.function([X,Y], T.mean(models), updates = updates)
		self.likelihood = theano.function([X,Y], -T.mean(models))
		
		self.entropy  =  (0.5 * param_length * (1 + np.log(2*np.pi)) + np.sum(np.log(self.prior_scale)))
	
		self.prior = theano.function([], T.mean(priors))
	def update(self, X,Y):
		de = []
		for echange in self.entropy_change:
			r0 = np.random.randn(self.param_length)
			r1 = echange(r0, X, Y)
			r2 = echange(r1, X, Y)						
			de.append(np.dot(r0, -2 * r0 + 3 * r1 - r2))				
		self.entropy+=np.mean(de)
		e2 =(0.5 * self.param_length * (1 + np.log(2*np.pi)) + self.param_length*(np.log(np.sqrt(2*self.lr.eval()))))
		self.entropy = 0.5*self.param_length*np.log(np.exp(2*self.entropy/self.param_length)+np.exp(2*e2/self.param_length))	
		return self._update(X,Y)
	
	def evidence(self,X,Y):
		like = self.likelihood(X,Y)
		prior = self.prior()
		entropy = self.entropy
		evidence =  prior +  like + entropy
		
		return evidence, (prior, like, entropy)
		
if __name__=='__main__':
	#1-layer nnet with 20 neurons 
	train_x = np.load('../../data/mini_train_x.npy')
	train_y = np.load('../../data/train_y.npy')
	neuron_num = 20
	lr =  1.0/train_x.shape[0]

	def builder(params, X_, Y_):
		def logsumexp(X, axis):
		    max_X = T.max(X, axis=axis, keepdims=True)
    		    return max_X + T.log(T.sum(T.exp(X - max_X), axis=axis, keepdims=True))

		W1 = params[:50*20].reshape((50,20))
		W2 = params[50*20:50*20+20*10].reshape((20,10))
		b1 = params[-20-10:-10]
		b2 = params[-10:]
		
		first_layer = T.dot(X_, W1) + b1
		second_layer = T.dot(T.tanh(first_layer),W2) + b2
		softmax_fn = second_layer - logsumexp(second_layer,1)
		cost = -T.sum(softmax_fn[T.arange(Y_.shape[0]), Y_])*train_x.shape[0]/X_.shape[0]

		return cost
	X_ = T.matrix()
	Y_ = T.ivector()
	lang_test = LangevinEstimator(50*20+20*10+20+10, [1.0],  X_, Y_, builder, multistart=1, lr=lr)
	elems = list(range(0,train_x.shape[0]))
	for i in range(0,1000):
		np.random.shuffle(elems)
		lang_test.lr.set_value(lr* (i+1)**(-0.51))

		if i%50==0:
			result, meta = lang_test.evidence(train_x,train_y)
			print i, result/train_x.shape[0], meta[0]/train_x.shape[0], meta[1]/train_x.shape[0], meta[2]/train_x.shape[0]

		lang_test.update(train_x[elems[:100]],train_y[elems[:100]])
		


				

