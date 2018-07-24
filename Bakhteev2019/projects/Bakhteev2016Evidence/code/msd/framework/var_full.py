import numpy as np
import theano
import theano.tensor as T
train_x = np.load('../../data/mini_train_x.npy')
train_y = np.load('../../data/train_y.npy')
np.random.seed(42)
multistart_num = 1
class_num = 10
batch_size = 100
lr = theano.shared(1.0/train_x.shape[0])
iter_num = 1000
from sklearn import linear_model
#for softmax logarithm

def logsumexp(X, axis):
    max_X = T.max(X)
    return max_X + T.log(T.sum(T.exp(X - max_X), axis=axis, keepdims=True))

result = []
class NetStructure:
	X_s = theano.shared(train_x)
	Y_s = theano.shared(train_y)
	I_s = T.ivector()
	def __init__(self, neuron_num):
		from theano.tensor.shared_randomstreams import RandomStreams
		self.rs = RandomStreams()
		init_prior_std1 =1.0
                init_prior_std2 = 1.0

		self.neuron_num = neuron_num
		param_num = train_x.shape[1] * neuron_num + neuron_num*class_num + class_num + neuron_num #total number of parameters
		self.multiplier_ =  [init_prior_std1]*(train_x.shape[1] * neuron_num ) + [init_prior_std2]*neuron_num*class_num+[init_prior_std1]*neuron_num+[init_prior_std2]*class_num
		self.rv_n = self.rs.normal((param_num,))

		self.params = theano.shared(np.random.normal(size=(param_num))* self.multiplier_) #note, multiplier is not a diagonal of covariance, it's a square root of it!	

		
		self.W1 = self.params[:train_x.shape[1] * neuron_num].reshape((train_x.shape[1], neuron_num))
		self.W2 = self.params[train_x.shape[1] * neuron_num:train_x.shape[1] * neuron_num+neuron_num*class_num].reshape((neuron_num, class_num))
		self.b1 = self.params[-neuron_num-class_num:-class_num].reshape((neuron_num, ))
		self.b2 = self.params[-class_num:].reshape(( class_num,))		
		self.first_layer = T.dot(NetStructure.X_s[NetStructure.I_s], self.W1)+self.b1
		self.second_layer = T.dot(T.tanh(self.first_layer), self.W2) + self.b2	
		#we don't need softmax during the training, but the log of softmax
		self.softmax_fn = self.second_layer - logsumexp(self.second_layer,1)
	        prior_std1 = init_prior_std1
                prior_std2 = init_prior_std2
                self.multiplier =  [prior_std1]*(train_x.shape[1] * neuron_num ) + [prior_std2]*neuron_num*class_num+[prior_std1]*neuron_num+[prior_std2]*class_num
		self.shared_mult = theano.shared(np.array(self.multiplier))
		self.neg_prior = 0.5 * param_num * np.log(2*np.pi) + T.sum(T.log(self.shared_mult)) + 0.5 * T.dot(self.params/self.shared_mult, self.params/self.shared_mult)	
		self.monitor_cost = -T.sum(self.softmax_fn[T.arange(NetStructure.Y_s[NetStructure.I_s].shape[0]), NetStructure.Y_s[NetStructure.I_s]])*train_x.shape[0]/NetStructure.X_s[NetStructure.I_s].shape[0]
		
		self.cost =self.monitor_cost + self.neg_prior
		self.g_p = T.grad(self.cost, self.params)
		self.r = T.vector()
		self.hvp = T.grad(T.dot(self.g_p, self.r), self.params)
		self.jvp = self.r - lr* self.hvp
		self.updates = [(self.params, self.params - (lr) * self.g_p+0*self.rv_n*T.sqrt(lr))]
		self.test = theano.function([NetStructure.I_s,self.r], T.dot(self.g_p, self.r), allow_input_downcast=True)
		self.hvp_f = theano.function([NetStructure.I_s,self.r], self.jvp, allow_input_downcast=True)
#the structure of the net: input - hidden layer (tanh) - softmax
plot_results = []
import time

for neuron_num in xrange(20, 101,5):# according to out article the minimum of neurons is class_num+1

		plot_results.append([])
		param_num = train_x.shape[1] * neuron_num + neuron_num*class_num + class_num + neuron_num 
		prior_std1 = 1.0
		prior_std2 = 1.0
		
		starts = []
		updates = []
		
			
		monitors = []
		train_likelihood_monitor = [] 
		for _ in range(0, multistart_num):
			starts.append(NetStructure(neuron_num))
			updates.extend(starts[-1].updates)
			monitors.append(-starts[-1].monitor_cost/train_x.shape[0] )	
			
			train_likelihood_monitor.append(-starts[-1].monitor_cost/train_x.shape[0])

		
		train = theano.function([NetStructure.I_s], monitors, updates= updates, allow_input_downcast=True)
		likelihood_monitor = 	theano.function([NetStructure.I_s], train_likelihood_monitor,  allow_input_downcast=True)
		#random indexes for batches	
		indexes = [i for i in xrange(0, train_x.shape[0])]
		#starting entropy
		           
		entropy =  (0.5 * param_num * (1 + np.log(2*np.pi)) + np.sum(np.log(starts[-1].multiplier_)))

		print neuron_num, 'starting'
		old_evidences = None 
		patience = 5
		all_evidences = [] 
		old_train_likelihood = -99999999999999999999
		mean_evs = []
		best = -999999999999999
		best_evs = []
		
		time_s = time.time()
		for i in xrange(0, iter_num):
			
			
			layer1, layer2 = [],[]
			for j in range(0, multistart_num):
				layer1.extend(starts[j].W1.eval().flatten().tolist())
				layer2.extend(starts[j].W2.eval().flatten().tolist())
				layer1.extend(starts[j].b1.eval().tolist())
				layer2.extend(starts[j].b2.eval().tolist())			
			
			np.random.shuffle(indexes)  
			              
			priors, likelihoods, entropies = [],[],[]
			
			for j in range(0, multistart_num):
				r0 =np.random.randn(param_num)#critical ~ 0.005
	
				r1 = starts[j].hvp_f(indexes[:batch_size], r0)#~0.003

				r2 = starts[j].hvp_f(indexes[:batch_size], r1)#~0.01							
				
				e1 =entropy+ np.dot(r0, -2 * r0 + 3 * r1 - r2)
				e2 =(0.5 * param_num * (1 + np.log(2*np.pi)) + param_num*(np.log(np.sqrt(lr.eval()))))
				
				entropies.append(e1)
				
			entropy = np.mean(entropies)
			#entropy = 0.5*param_num*np.log(np.exp(2*entropy/param_num)+np.exp(2*e2/param_num))	
			
			
			if i%50==0:
				for j in range(0, multistart_num):
					neg_prior = 0.5 * param_num * np.log(2*np.pi) + np.sum(np.log(starts[j].multiplier)) + 0.5 * np.dot(starts[j].params.eval()/starts[j].multiplier, starts[j].params.eval()/starts[j].multiplier)
					
					priors.append(-neg_prior/train_x.shape[0] )
				likelihoods = likelihood_monitor(indexes)
				evidences = np.array(priors) +  np.array(likelihoods)+ entropy/train_x.shape[0]							
				print neuron_num,i, evidences, np.mean(evidences)
				print 'prior likelihood entropy',np.mean(priors),  np.mean(likelihoods),  np.mean(entropies) /train_x.shape[0]

				#print e1, e2, delta_e
			train(indexes[:batch_size])#~0.01


		print time.time() - time_s
		priors = []
		for j in range(0, multistart_num):
			neg_prior = 0.5 * param_num * np.log(2*np.pi) + np.sum(np.log(starts[j].multiplier)) + 0.5 * np.dot(starts[j].params.eval()/starts[j].multiplier, starts[j].params.eval()/starts[j].multiplier)
			priors.append(-neg_prior/train_x.shape[0] )	
		likelihoods = likelihood_monitor(indexes)
		evidences = np.array(priors) +  np.array(likelihoods)+ entropy/train_x.shape[0]
		exit(1)
		
		#print result

