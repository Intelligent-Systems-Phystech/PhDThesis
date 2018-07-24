import sys
sys.path.append('.')
import theano
import theano.tensor as T
import numpy as np
import random 
from structures import HyperparameterOptimization
import random 
import time
from scipy.optimize import minimize
import gc
def hoag_optimize(trainig_criterion, model_constructor, param_optimizer,  trial_num , batch_size, batch_size2, train_iteration_num, X_data, Y_data, hyperparams,  internal_optimize_learning_rate=10**(-5), internal_optimize_eps = 0.98, limits=None,  max_abs_err = 10**10, 
    lr=0.0,   verbose=0):
   
   
    history = []  
    dataset_size  = np.array(X_data).shape[0]
    X_datas = theano.shared(X_data)
    Y_datas = theano.shared(Y_data)
    if len(hyperparams)>1:
        raise NotImplementedError('Sorry, not implemented: num of hyperparams > 1')  
    
    training_procedure = trainig_criterion( model_constructor, param_optimizer,X_data, Y_data )
    k = len(training_procedure.models)
    
    
    """
    1. solve
    2. make hessian optimization
    3. make derivatives
    4. correct
    """
    Xs = [T.matrix() for _ in xrange(k)]
    Ys = [T.vector(dtype=Y_data.dtype) for _ in xrange(k)]
    indices = [T.ivector() for _ in xrange(k)]
    Xs2 = [T.matrix() for _ in xrange(k)]
    Ys2 = [T.vector(dtype=Y_data.dtype) for _ in xrange(k)]
    indices2 = [T.ivector() for _ in xrange(k)]


   
    costs = []
    givens = []

    
    costs_valid = []
    all_params = []
    for X,Y, index,  model, X2, Y2, index2 in zip(Xs, Ys, indices,  training_procedure.models,Xs2, Ys2, indices2):
        train_cost = model.cost(X, Y)
       
        all_params.append(model.params)
       
        
        validation_cost = model.validation(X2, Y2)


        givens.append((X,X_datas[index]))
        givens.append((Y,Y_datas[index]))
        givens.append((X2,X_datas[index2]))
        givens.append((Y2,Y_datas[index2]))
       
        costs.append(-train_cost) #using negative for article correspondence
        costs_valid.append(-validation_cost) #using negative for article correspondence
    
    valid_cost = T.mean(costs_valid)
    cost = T.mean(costs)
    
    q = [theano.shared(np.zeros(len(all_params[0].eval())).astype(theano.config.floatX)) for _ in xrange(k)]

    
    h_2 = T.grad(cost,all_params)
    Hq = T.Rop(h_2, all_params, q) #submodels are independent
    g_1 = T.grad(valid_cost, all_params) #test: 2x2
    g_2 = T.grad(valid_cost, hyperparams, disconnected_inputs='ignore')
    
    g_2 = theano.function(indices2, g_2, givens=givens, on_unused_input='ignore')
    h_1 = T.grad(cost, all_params) #test: 2x2
        
    #h_1_2s = [theano.gradient.jacobian(h_1_, hyperparams, disconnected_inputs='ignore' ) for h_1_, q_ in zip(h_1, q)]
    #h_1_2s_conc = T.concatenate([h_[0] for h_ in h_1_2s], axis=0) #test expecting: 4x2
    
    #h_1_2_q = theano.function(indices, T.dot(h_1_2s_conc.T, T.concatenate(q)),  givens=givens)
    
    h_1_2s = T.Lop(T.concatenate(h_1), hyperparams, T.concatenate(q))
    h_1_2_q = theano.function(indices,  h_1_2s[0],  givens=givens)
    internal_cost = T.mean((T.concatenate(Hq) - T.concatenate(g_1))**2)
    internal_grad = T.grad(internal_cost, q)
    updates = [(q_, q_-internal_optimize_learning_rate*internal_grad_) for q_, internal_grad_ in zip(q, internal_grad)]
    
    internal_update = theano.function(indices+indices2, internal_cost, givens=givens, updates= updates+model.train_updates)
    internal_monitor = theano.function(indices+indices2, internal_cost, givens=givens)
    #internal_update = theano.function(indices+indices2, Hq, givens=givens, on_unused_input='ignore')
    for trial in xrange(trial_num): 
        gc.collect()
        for m in    training_procedure.models:
            m.respawn()
        if verbose>=0 :
            print 'trial ', trial
        for i in xrange(train_iteration_num):
            
            res  = training_procedure.do_train()
            
            if verbose>=0 and (verbose==0 or i%verbose==0):
                print 'iteration {0}, internal loss={1}'.format(str(i), str(res))
        
        valid_score = training_procedure.do_validation()
        history.append(([h.eval() for h in hyperparams], valid_score))
        if verbose>=0:
            
            print 'validation score: ', valid_score
        
        if verbose>=0:
            print 'internal optimization'


        err = None
        rel_err = -1# -1
        attemp_num = 10
        while rel_err<internal_optimize_eps:
            
            sample1 = [random.sample(ti,batch_size2) for ti in training_procedure.train_indices]
            sample2 = [random.sample(vi,batch_size2) for vi in training_procedure.validation_indices]

            err_new = internal_update(*(sample1+sample2))
            gc.collect()            
            if err is not None:
                rel_err = min(err_new,err)/max(err_new,err)
            #qs = [q_.eval() for q_ in q]
            
            #if attemp_num> 0 and (np.isnan(np.mean(qs)) or np.isinf(np.mean(qs)) or (err_new > max_abs_err)) :
            #    attemp_num-=1
                
            #    print 'bad internal learning rate', err_new, err, np.mean(qs)
            #    if internal_optimize_learning_rate/10 > 0:
            #        internal_optimize_learning_rate = internal_optimize_learning_rate/10
            #        for q_ in q:
            #            q_.set_value(np.zeros(len(all_params[0].eval())))
            #            err = None
            #            print 'updating learning rate', internal_optimize_learning_rate
            err = err_new#abs(err_new - err)/(err + err_new)
        
            if verbose>=0:
                print rel_err, err

        sample_t = [random.sample(ti, batch_size2) for ti in training_procedure.train_indices]
        sample_v = [random.sample(vi, batch_size2) for vi in training_procedure.validation_indices]
        time_s = time.time()
        #print len(g_2(*sample1))
        #print  len(h_1_2_q(*sample2))
        grads =  g_2(*sample_v) - h_1_2_q(*sample_t) 
        #print 'TIME', time_s - time.time()
        #g_2(*training_procedure.validation_indices)##g_2(*training_procedure.validation_indices)# - h_1_2_q(*training_procedure.train_indices)
        #print 'grads', grads
        #print h_1_2_q(*training_procedure.train_indices) 
        good_update = False 
        attemp_num = 10
        ####TODO
        while not good_update and attemp_num>0:
            good_update = True   

            if limits:
                old_values = [h.eval() for h in hyperparams]    
            for h,l,g in zip(hyperparams, lr,grads):
                #print (g).dtype, (h.eval()).dtype, type(l)
                h.set_value(h.eval() - l * g)            
            if limits:
                h_id = -1
                for h, l in zip(hyperparams, limits):
                    h_id+=1
                    he = h.eval()
                    if np.max((he))>l[1] or np.min((he))<l[0] or np.isnan(np.max(he)) or np.isinf(np.max(he)):
                        print 'bad hyperparam update'
                        print he,' vs limit ',l 
                        if np.isnan(np.max(he)) or np.isinf(np.max(he)):
                            h.set_value(o)
                        else:
                            h.set_value(np.minimum(l[1], np.maximum(l[0], he)))

                        for h2,o in zip(hyperparams, old_values):
                            print 'returning value', o
                            h2.set_value(o)
                        lr[h_id]=  lr[h_id] / 10.0
                        
                        print 'new lr', lr
                        attemp_num -= 1
                        good_update = False 
                               



        
        if verbose>=0:
            print 'hypergrads', grads
    return HyperparameterOptimization(best_values=history[-1][0], history=history)
        
if __name__=='__main__':
    from generic.optimizer import gd_optimizer
    from pyfos.models.var_feedforward import build_var_feedforward
    from tc.cv import  cv_tc
    from functools import partial
    matrix = np.load('../../data/matrix.npy')
    X, Y = np.load('../../data/linearx.npy'), np.load('../../data/lineary.npy')
    X_train = X[:100]
    Y_train = Y[:100]
    X_test = X[100:]
    Y_test = Y[100:]
    lr = theano.shared(10**(-3))
    log_alphas = theano.shared(np.array([.0, .0]))

    optimizer = partial(gd_optimizer, learning_rate=lr)
    model_build = partial(build_feedforward, structure = [2,1], nonlinearity=lambda x:x, log_alphas=log_alphas, bias=False)


        
    hoag_optimize(partial(cv_tc, k =3,  batch_size=75),
                model_build,  
                optimizer,
                  25, 75,
                 100, 
                X_train, Y_train, [log_alphas], lr=[0.01],verbose=10)#10**(-7), verbose=1)
