import theano
import theano.tensor as T
import numpy as np
import random 
import sys
import time
sys.path.append('.')


from structures import HyperparameterOptimization
def greed_optimize(trainig_criterion, model_constructor, param_optimizer,   trial_num , batch_size,  train_iteration_num, X_data, Y_data, hyperparams, limits=None,   lr=0.0, 
verbose=0):
 
    history = [] 
    
    dataset_size  = np.array(X_data).shape[0]
    X_datas = theano.shared(X_data)
    Y_datas = theano.shared(Y_data)
    training_procedure = trainig_criterion( model_constructor, param_optimizer,X_data, Y_data )
    #training_procedure.models[0].params.set_value( [5.98813441,  18.23289179,  -1.43155485,  11.13149291])

    k = len(training_procedure.models)
    Xs = [T.matrix() for _ in xrange(k)]
    Ys = [T.vector(dtype=Y_data.dtype) for _ in xrange(k)]
    indices = [T.ivector() for _ in xrange(k)]
    Xs2 = [T.matrix() for _ in xrange(k)]
    Ys2 = [T.vector(dtype=Y_data.dtype) for _ in xrange(k)]
    indices2 = [T.ivector() for _ in xrange(k)]
    costs = []
    givens = []
    for X,Y, index, X2, Y2, index2, model in zip(Xs, Ys, indices, Xs2, Ys2, indices2, training_procedure.models ):
        train_cost = model.cost(X, Y)
        updates = param_optimizer(train_cost, model.params)

        new_params = [u[1] for u in updates]
        if len(new_params)!=1:
            raise NotImplementedError('model with multiple params')
        else:
            new_params = new_params[0]

        new_model = model_constructor(dataset_size=dataset_size, params=new_params)

        validation_cost = new_model.validation(X2, Y2)


        givens.append((X,X_datas[index]))
        givens.append((Y,Y_datas[index]))
        givens.append((X2,X_datas[index2]))
        givens.append((Y2,Y_datas[index2]))
        costs.append( validation_cost)
    valid_cost = T.mean(costs)
    grad = T.grad(valid_cost, hyperparams)
    try:
        len(lr)
    except:
        lr = [lr]*len(hyperparams) #TODO: refactor :)))
    lr = [theano.shared(l) for l in lr]
    updates = [(h, h + g*l) for h,g,l in zip(hyperparams, grad, lr)]
    hypertrain = theano.function(indices+indices2, valid_cost, updates=updates, givens=givens)
    show_grads = theano.function(indices+indices2, grad, givens=givens)
    def hypertrain_callback(inds):
     
        good_update = False 
        attemp_num = 10
        while not good_update and attemp_num>0:
            good_update = True
            if limits:
                old_values = [h.eval() for h in hyperparams]  
            valids = []
            for valid_subind in training_procedure.validation_indices:
                try:
                    sample = random.sample(valid_subind, batch_size)
                except:
                    sample = valid_subind
            
                valids.append(sample)
         
            res = hypertrain(*(inds + valids))
            if limits:
                
                #grad_debug =  show_grads(*(inds + training_procedure.validation_indices))
                h_id = -1
                for h, l in zip(hyperparams, limits):
                    h_id += 1
                    he = h.eval()

                    if np.max((he))>l[1] or np.min((he))<l[0] or np.isnan(np.max(he)) or np.isinf(np.max(he)):
                        print 'bad hyperparam update'
                        attemp_num-=1
                        print he,' vs limit ',l 
                        if np.isnan(np.max(he)) or np.isinf(np.max(he)):
                            h.set_value(old_values[h_id])
                        else:
                            h.set_value(np.maximum(np.minimum(he, l[1]), l[0]))
                       
                        lr[h_id].set_value(lr[h_id].eval()/10.0)
                    
        
        return res
    trial = -1 
    time_s = time.time()
    for i in xrange(train_iteration_num):

            res = training_procedure.do_train()
            if verbose>=0 and (verbose==0 or i%verbose==0):

                print 'iteration {0}, internal loss={1}, time={2}'.format(str(i), str(res), str(time.time() - time_s))
                time_s = time.time()

    score = training_procedure.do_validation()

    history.append(([h.eval() for h in hyperparams], score))

    for trial in range(trial_num):
        for m in training_procedure.models:
            m.respawn()
        if verbose>=0:
            print 'trial', trial, [h.eval() for h in hyperparams]
        time_s = time.time()
        #time.sleep(15)        
        for i in xrange(train_iteration_num):

            res, call_res = training_procedure.do_train(callback=hypertrain_callback)
            if verbose>=0 and (verbose==0 or i%verbose==0):
                #print 'param',training_procedure.models[0].params.eval()
                print 'iteration {0}, internal loss={1} hyperparam loss={2} time = {3}'.format(str(i), str(res), str(call_res), str(time.time() - time_s))
                time_s = time.time()

        score = training_procedure.do_validation()
        if verbose>=0:
            print 'validation {0}'.format( str(score))
        history.append(([h.eval() for h in hyperparams], score))
            
       
    return HyperparameterOptimization(best_values=[h.eval() for h in hyperparams], history=history)
        

if __name__=='__main__':
    import numpy as np
    import sys
    sys.path.append('.')
    sys.path.append('..')

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
    lr = theano.shared(10**(-3))
    alphas = theano.shared(np.array([1.0, 1.0]))

    optimizer = partial(gd_optimizer, learning_rate=lr)
    model_build = partial(build_var_feedforward, structure = [2,1], nonlinearity=lambda x:x, log_alphas=alphas, bias=False, param_pool_size=100)

    hyp_lr_range = [0.01]
    greed_optimize(partial(simple_tc,   batch_size=100), model_build, optimizer,50, 100,  X_train, Y_train,  [alphas] , 
               lr=hyp_lr_range, verbose=-1, limits=[[-2,12.0]]
    )

    
    
