import sys
sys.path.append('.')
import numpy as np
import theano
import theano.tensor as T
from generic.regularizers import KLD
from structures import Model
"""
There are some troubles with jacobian + random, so now we are using precomputed random numbers
See:
https://groups.google.com/forum/#!topic/theano-users/X40EeXTOSas
if theano.config.device=='cpu':
    from theano.tensor.shared_randomstreams import RandomStreams
else:
    from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
"""
from theano.tensor.shared_randomstreams import RandomStreams
def logsumexp(X, axis):
        max_X = T.max(X, axis=axis, keepdims=True)
        return max_X + T.log(T.sum(T.exp(X - max_X), axis=axis, keepdims=True))

def build_simplified_var_feedforward(structure,  params=None, init_sigmas = None, nonlinearity = T.tanh, use_softmax = False, log_alphas = theano.shared(np.array([0.0]).astype(theano.config.floatX)), dataset_size=1, megabatch_size=50,  bias=True, param_pool_size=1, log_var_sigmas=None):

    if bias:
        bias_coef =1                                                                                                                                                                                                                                                                
    else:
        bias_coef = 0

    _param_num = 0
    if init_sigmas is None:
        init_sigmas = [0] * len(structure)

    _multiplier = []
    for i in xrange(len(structure)-1):
        _param_num += (structure[i]*structure[i+1] + structure[i+1]*bias_coef)*2
        _multiplier.extend([init_sigmas[i]]*(structure[i]*structure[i+1]) + [0]*(structure[i+1]*bias_coef))
    
    if log_var_sigmas is None:
        log_var_sigmas = np.array([-2.0] * (_param_num/2)).astype(theano.config.floatX)

    use_softmax = use_softmax
    nonlinearity = nonlinearity
    _structure = structure
    if not params:      
        params =  (np.random.randn(_param_num/2)*_multiplier).tolist()+log_var_sigmas.tolist()
        params = theano.shared(np.array(params).astype(theano.config.floatX))
    def respawn():
        
        p_values = (np.random.randn(_param_num/2)*_multiplier).astype(theano.config.floatX).tolist()+log_var_sigmas.tolist()
        #print '111', p_values
        params.set_value(np.array(p_values).astype(theano.config.floatX))

    #variational vs prior
    #regularized_sigmas = T.maximum(np.ones(_param_num/2)*(-10), 2*log_var_sigmas)
    _KLD = KLD(params[:_param_num/2], np.zeros(_param_num/2).astype(theano.config.floatX), 2*log_var_sigmas, 2*log_alphas)#regularized_sigmas, 2*(log_alphas))
    """
    if len(alphas.eval())==1:
        prior = gaus_prior(params, alphas=alphas, gaus_type='scalar')
    else:
        prior = gaus_prior(params, alphas=alphas, gaus_type='diagonal')
    """
    #srng = RandomStreams()
    #randoms =  theano.shared(np.random.randn(param_pool_size,_param_num/2).astype(theano.config.floatX))
    #expsigm = T.exp(params[_param_num/2:])
    #means = params[:_param_num/2]
    #random_noise = randoms*expsigm#T.tile(expsigm, (25,1))
    #sigmas = params[_param_num/2:]#T.tile(params[:_param_num/2], (25,1))
    #random_pool = random_noise    +  means

    randoms =  theano.shared(np.random.randn(param_pool_size,_param_num/2).astype(theano.config.floatX))
    expsigm = T.exp(params[_param_num/2:])
    
    random_noise = randoms*expsigm#T.tile(expsigm, (25,1))
    means = params[:_param_num/2]#T.tile(params[:_param_num/2], (25,1))
    random_pool = random_noise    +  means

    
    """ 
    random_pool_selector = srng.choice(size=(X.shape[0]), a=2, replace=True, p=None, ndim=None, dtype='int64')
    
    """
    Ws = []
    bs = []
    Ws_var = []
    bs_var = []
    Ws_noise = []
    bs_noise = []
    offset = 0
    structure = _structure
    
    for i in xrange(len(_structure)-1):
        W = params[offset:offset+structure[i]*structure[i+1]].reshape((structure[i], structure[i+1]))
        W_var = expsigm[offset:offset+structure[i]*structure[i+1]].reshape((structure[i], structure[i+1]))
        #W_var=  random_pool[:, offset:offset+structure[i]*structure[i+1]].reshape((param_pool_size, structure[i], structure[i+1]))
        
        W_noise=  randoms[:, offset:offset+structure[i]*structure[i+1]].reshape((param_pool_size, structure[i], structure[i+1]))
        if bias:
            b = params[offset+structure[i]*structure[i+1]:offset + structure[i]*structure[i+1] + structure[i+1] ]
            b_var = expsigm[offset+structure[i]*structure[i+1]:offset + structure[i]*structure[i+1] + structure[i+1] ]
            #b_var = random_pool[:, offset+structure[i]*structure[i+1]:offset + structure[i]*structure[i+1] + structure[i+1] ].reshape((param_pool_size, structure[i+1] ))
            
            b_noise = randoms[:, offset+structure[i]*structure[i+1]:offset + structure[i]*structure[i+1] + structure[i+1] ].reshape((param_pool_size, structure[i+1] ))
        Ws.append(W)
        
        
        Ws_var.append((W_var))
        Ws_noise.append(W_noise)
        offset+=structure[i]*structure[i+1]
        if bias:
            bs.append(b)
            bs_var.append((b_var))
            bs_noise.append(b_noise)
            offset+=structure[i+1]
    
    pool_ind = theano.shared(0)
    updates = [(pool_ind, T.mod(pool_ind+1, param_pool_size))]
    
    def simple_build_likelihood(X, Y, ind=pool_ind):
       
        output = X
       
        
        
        for i in xrange(len(_structure)-1):
                if i== len(structure)-2:
                    nonlin = lambda x:x
                else:
                    nonlin = nonlinearity
                W =Ws[i] +  (Ws_var[i])*Ws_noise[i][ind]
                #W = Ws_var[i][ind]
                #W = T.tile(Ws_var[i], (mult_factor, 1, 1))[:X.shape[0]]#[random_pool_selector]
                if bias:
                    #b = T.tile(bs_var[i], (mult_factor, 1))[:X.shape[0]]
                    #b = bs_var[i][ind]
                    b =bs[i] +  (bs_var[i])*bs_noise[i][ind]
                    output = nonlin(T.dot(output, W)  + b)
                else:
                    output = nonlin(T.dot(output, W) )
        if use_softmax:
            softmax_fn = output - logsumexp(output,1)
            log_likelihood = T.sum(softmax_fn[T.arange(Y.shape[0]), Y])            
        else:
            log_likelihood  = -0.5*T.sum((output.T - Y)**2) - np.log(2*np.pi)*(X.shape[0]/2).astype(theano.config.floatX)  
        
        return log_likelihood
    
    
        
    def build_cost(X,Y):
        log_likelihood  = simple_build_likelihood(X,Y)
       
        return log_likelihood*dataset_size/X.shape[0]-_KLD
    
    def build_validation(X, Y):
        def myscanfunc(ind):
            X_ = X[ind*megabatch_size:(ind+1)*megabatch_size]
            Y_ = Y[ind*megabatch_size:(ind+1)*megabatch_size]
            return simple_build_likelihood(X_, Y_, ind=T.mod(pool_ind+ind, param_pool_size))
        result = theano.map (myscanfunc, sequences=[T.arange(T.max([1, X.shape[0]/megabatch_size]))]) [0]
        return T.sum(result)*dataset_size/X.shape[0]-_KLD
        
    def build_predict(X):
        output = X
        for i in xrange(len(_structure)-1):
            if i== len(structure)-2:
                    nonlin = lambda x:x
            else:
                    nonlin = nonlinearity
            if bias:
                output = nonlin(T.dot(output, Ws[i]) + bs[i])
            else:
                output = nonlin(T.dot(output, Ws[i]))
        if use_softmax:
            predict =  T.argmax(output, axis=1)
        else:
            predict = output


        return predict
        
    return Model(cost =  build_cost, validation=build_validation, params=params,  predict=build_predict, respawn=respawn,
                train_updates=updates)

if __name__=='__main__':
    #test regression
    X = T.matrix()
    Y = T.vector()
    
    model = build_var_feedforward([2,1], nonlinearity=lambda x:x, dataset_size=2, param_pool_size=10, log_alphas=np.array([0.0, 0.0, 0.0]))
    
    model.params.set_value([1,2,3, -10.0, -10.0, -10.0])
    
    predict =theano.function([X], model.predict_var(X))
    
    assert np.linalg.norm(predict([[4,5],[6,7]]) - [[17],[23]]) == 0
        
    
    cost = theano.function([X, Y], model.cost(X,Y),on_unused_input='ignore')
    
    #assert cost([[4,5],[6,7]], [0,0])!=cost([[4,5],[6,7]], [0,0])
    
    assert abs(cost([[4,5],[6,7]], [0,0]) - (-0.5*(17**2+23**2) - np.log(np.pi*2) - 7))<0.5
    
