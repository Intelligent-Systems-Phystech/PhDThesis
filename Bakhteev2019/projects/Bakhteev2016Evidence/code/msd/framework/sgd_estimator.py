from abstract_estimator import AbstractEstimator
import theano 
import theano.tensor as T
import numpy as np
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
np.random.seed(42)	
class SGDEstimator(AbstractEstimator):
	def __init__(self,  param_length, param_scale,  X, Y, ind, builder, **kwargs):
		self.multistart = kwargs.get('multistart',1)
		self.lr =  kwargs.get('lr',0.001)
		self.prior_scale =np.array(kwargs.get('prior_scale',param_scale)).astype(np.float32)
		self.param_length = param_length
		models = []
		updates = []
		srng = RandomStreams()
		self.entropy_changes = []
		self.input_vectors = []
		priors = []
		self.params =[]
		
		for i in xrange(0, self.multistart):
			r = T.vector()
			params = theano.shared((np.random.randn(param_length)*np.array(param_scale)).astype(np.float32))
			neg_prior = 0.5 * param_length * np.log(2*np.pi) + np.sum(np.log(self.prior_scale).astype(np.float32)) + 0.5 * T.dot(params/self.prior_scale, params/self.prior_scale)
			model = builder(params, X, Y)
			cost = model + neg_prior
			grad = T.grad(cost, params)
			def jvp(vector):				
				hvp = T.grad(T.dot(grad, vector), params)
				jvp = vector - self.lr * hvp 
				return jvp 
			
			
			r1 = jvp(r)
			#e_change = theano.function([r,ind], r1)
			self.entropy_changes.append(r1)
			self.input_vectors.append(r)
			self.params.append(params)
			updates.append((params, params - self.lr *  grad))
			models.append(model)
			
			priors.append(-neg_prior)
		self.entropy_change = theano.function([ind]+self.input_vectors, self.entropy_changes)
		self._update = theano.function([ind], T.mean(models), updates = updates)
		self.likelihood = theano.function([ind], -T.mean(models))
		
		self.entropy  =  (0.5 * param_length * (1 + np.log(2*np.pi)) + np.sum(np.log(param_scale)))
		self.prior = theano.function([], T.mean(priors))
	def update(self, ind):
		
		de = []
		input_vectors = []
		for i, echange in enumerate(self.input_vectors):
			
			input_vectors.append(np.random.randn(self.param_length).astype(np.float32))
		r1_all = self.entropy_change(ind,*input_vectors)#echange(r0, ind)
		r2_all = self.entropy_change(ind,*r1_all)	
		for r0, r1, r2 in zip(input_vectors, r1_all, r2_all):				
			de.append(np.dot(r0, -2 * r0 + 3 * r1 - r2))
	
		self.entropy+=np.mean(de)
		
		return self._update(ind)
	
	def evidence(self,ind):
	
		like = self.likelihood(ind)
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
	sgd_test = SGDEstimator(50*20+20*10+20+10, [1.0],  X_, Y_, builder, multistart=1, lr=lr)
	elems = list(range(0,train_x.shape[0]))
	for i in range(0,1000):
		np.random.shuffle(elems)

		if i%50==0:
			result, meta = sgd_test.evidence(train_x,train_y)
			print i, result/train_x.shape[0], meta[0]/train_x.shape[0], meta[1]/train_x.shape[0], meta[2]/train_x.shape[0]

		sgd_test.update(train_x[elems[:100]],train_y[elems[:100]])
		


				

