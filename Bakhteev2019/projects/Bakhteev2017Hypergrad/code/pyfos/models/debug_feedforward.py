import sys
sys.path.append('.')
import numpy as np
import theano
import theano.tensor as T
from generic.regularizers import gaus_prior
from structures import Model

def logsumexp(X, axis):
        max_X = T.max(X, axis=axis, keepdims=True)
        return max_X + T.log(T.sum(T.exp(X - max_X), axis=axis, keepdims=True))

def build_feedforward(structure,  params=None, init_sigmas = None, nonlinearity = T.tanh, use_softmax = False, log_alphas = theano.shared(np.array([0.0])), dataset_size=1, bias=True):
    if bias:
        bias_coef = 1
    else:
        bias_coef = 0

    _param_num = 0
    if init_sigmas is None:
        init_sigmas = [0] * len(structure)
    if len(init_sigmas) == 0:
        init_sigmas = [init_sigmas[0]]*len(structure)
    _multiplier = []
    for i in xrange(len(structure)-1):
        _param_num += structure[i]*structure[i+1] + structure[i+1]*bias_coef
        _multiplier.extend([init_sigmas[i]]*(structure[i]*structure[i+1]) + [0]*(structure[i+1]*bias_coef))
    
    use_softmax = use_softmax
    nonlinearity = nonlinearity
    _structure = structure
    _multiplier = np.array(_multiplier).astype(theano.config.floatX)

    if not params:       
        
        params = theano.shared(np.random.randn(_param_num).astype(theano.config.floatX) *_multiplier )

    def respawn():
        params.set_value(np.random.randn(_param_num).astype(theano.config.floatX)*_multiplier)

    prior = T.dot(params,  params.T)* log_alphas[0]
    Ws = []
    bs = []
    offset = 0
    structure = _structure

    for i in xrange(len(_structure)-1):
        
        W = params[offset:offset+structure[i]*structure[i+1]].reshape((structure[i], structure[i+1]))
        if bias:
            b = params[offset+structure[i]*structure[i+1]:offset + structure[i]*structure[i+1] + structure[i+1] ]
        Ws.append(W)
        if bias:
            bs.append(b)
        offset+= structure[i]*structure[i+1]
        if bias:
            offset+=structure[i+1]
    def build_cost(X,Y):
        outputs = [X]
        for i in xrange(len(_structure)-1):
            if i < len(_structure)-2:
                nonl = nonlinearity
            else:
                nonl = lambda x: x
            if bias:
                outputs.append( nonl(T.dot(outputs[-1], Ws[i]) + bs[i]))
            else:
                outputs.append(nonl(T.dot(outputs[-1], Ws[i])))
        
        if use_softmax:
            softmax_fn = outputs[-1] - logsumexp(outputs[-1],1)
            log_likelihood = T.mean(softmax_fn[T.arange(Y.shape[0]), Y])            
        else:
            log_likelihood  = -0.5*T.sum((outputs[-1].T - Y)**2) - np.log(2*np.pi)*(X.shape[1]*X.shape[0]*1.0/2)      
        return log_likelihood+prior


    def build_validation(X,Y):
        outputs = [X]
        for i in xrange(len(_structure)-1):
            if i < len(_structure)-2:
                nonl = nonlinearity
            else:
                nonl = lambda x: x
            if bias:
                outputs.append( nonl(T.dot(outputs[-1], Ws[i]) + bs[i]))
            else:
                outputs.append(nonl(T.dot(outputs[-1], Ws[i])))
        
        if use_softmax:
            softmax_fn = outputs[-1] - logsumexp(outputs[-1],1)
            log_likelihood = T.mean(softmax_fn[T.arange(Y.shape[0]), Y])            
        else:
            log_likelihood  = -0.5*T.sum((outputs[-1].T - Y)**2) - np.log(2*np.pi)*(X.shape[1]*X.shape[0]*1.0/2)            
        #print dataset_size
        return log_likelihood

    def build_predict(X):
        outputs = [X]
        for i in xrange(len(_structure)-1):
            if i < len(_structure)-2:
                nonl = nonlinearity
            else:
                nonl = lambda x: x

            if bias:
                outputs.append(nonl(T.dot(outputs[-1], Ws[i]) + bs[i]))
            else:
                outputs.append( nonl(T.dot(outputs[-1], Ws[i])))

        if use_softmax:
            predict =  T.argmax(outputs[-1], axis=1)
        else:
            predict = outputs[-1]


        return predict
        
    return Model(cost =  build_cost,  validation= build_validation , params=params,  predict=build_predict, respawn=respawn)

if __name__=='__main__':
    """
    #test regression
    X = T.matrix()
    Y = T.vector()
    model = build_feedforward([2,1], nonlinearity=lambda x:x, dataset_size=2)
    model.params.set_value([1,2,3])
    predict =theano.function([X], model.predict_var(X))
    cost = theano.function([X, Y], model.cost(X,Y))
    assert np.linalg.norm(predict([[4,5],[6,7]]) - [[17],[23]]) == 0
    #print cost([[4,5],[6,7]], [0,0])
    #print -0.5*(17**2+23**2) - np.log(np.pi*2) - np.log(2*np.pi) * 1.5  -0.5*np.dot([1,2,3], [1,2,3])
    assert cost([[4,5],[6,7]], [0,0])  == -0.5*(17**2+23**2) - np.log(np.pi*2) - np.log(2*np.pi) * 1.5  -0.5*np.dot([1,2,3], [1,2,3])
    # test classification
    X = T.matrix()
    Y = T.ivector()
    model = build_feedforward( [2,2], dataset_size=2,  nonlinearity=lambda x:x, use_softmax=True)
    model.params.set_value([1, 2, 3,4 , 5,6])
    predict =theano.function([X], model.predict_var(X))
    assert np.linalg.norm(predict([[4,5],[6,7]]) - [[1],[1]]) == 0
    cost = theano.function([X, Y], model.cost(X,Y), on_unused_input='ignore')
    assert np.linalg.norm(cost([[4,5],[6,7]], [0,0]) - (-24.0  - np.log(2*np.pi) * 3  -0.5*np.dot([1,2,3,4,5,6], [1,2,3,4,5,6]))) < 0.01
    """
    X = T.matrix()
    Y = T.matrix()
    data = np.random.randn(25, 306), np.random.randn(25, 3)
    
    #model = build_feedforward([306, 10, ], nonlinearity=lambda x:x, dataset_size=2)
  
    cost = theano.function([X, Y], model.cost(X,Y), on_unused_input='ignore')
    print cost(data[0], data[1]).shape

    
