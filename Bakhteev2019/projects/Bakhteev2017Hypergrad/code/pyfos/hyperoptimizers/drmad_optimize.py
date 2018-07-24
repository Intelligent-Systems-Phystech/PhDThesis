import sys
sys.path.append('.')
import theano
import theano.tensor as T
import numpy as np
import random 
from structures import HyperparameterOptimization
import random 
import time



def drmad_optimize(trainig_criterion, model_constructor, param_optimizer,  trial_num , batch_size, train_iteration_num, X_data, Y_data, hyperparams, learning_rate_param,  reverse_start =0, reverse_step=1, 
    lr=0.0, lr_for_learning_rate=0.0, limits=None, lr_limits=None, verbose=0, use_hessian=True):
    optimize_learning_rate = lr_for_learning_rate != 0.0


    history = []  
    dataset_size  = np.array(X_data).shape[0]
    X_data_s = theano.shared(X_data)
    Y_data_s = theano.shared(Y_data)  
    time_s = time.time()
    training_procedure = trainig_criterion( model_constructor, param_optimizer,X_data, Y_data )
  
    k = len(training_procedure.models)

    Xs = [T.matrix() for _ in xrange(k)]
    Ys = [T.vector(dtype=Y_data.dtype) for _ in xrange(k)]
    indices = [T.ivector() for _ in xrange(k)]
    model_size = training_procedure.models[0].params.eval().shape
    W0_tensor = [theano.shared(np.zeros(model_size).astype(theano.config.floatX)) for _ in xrange(k)]
    beta = T.scalar()
    param_tensors = [W0*beta + (1-beta)*model.params for W0, model in zip(W0_tensor,training_procedure.models )]
   
    costs = []
    givens = []

    
    costs_valid = []
    all_params = []
    for X,Y, index,  model, p_tens in zip(Xs, Ys, indices,  training_procedure.models, param_tensors ):
        train_cost = -model.cost(X, Y)
        valid_cost = -model.validation(X, Y)
        all_params.append(model.params)

        
        

        givens.append((X,X_data_s[index]))
        givens.append((Y,Y_data_s[index]))
        givens.append((model.params,p_tens))
        costs_valid.append(valid_cost)
        costs.append(train_cost)
    
    valid = T.mean(costs_valid)
    cost = T.mean(costs)

    valid_grad_params = T.grad(valid, all_params)
    cost_grad_params = T.grad(cost, all_params)

    cost_grad_hyper = T.grad(cost, hyperparams)
    valid_grad_hyper = T.grad(valid, hyperparams, disconnected_inputs='ignore')

    cost_grad_params_hyper_by_models = []
    for grad in cost_grad_params:
        cost_grad_params_hyper_by_models.append(theano.gradient.jacobian(grad, hyperparams))
    
    vgp = theano.function(indices+[beta], valid_grad_params, givens=givens)
    vgh = theano.function(indices+[beta], valid_grad_hyper, givens=givens, on_unused_input='ignore')

    cgp = theano.function(indices+[beta], cost_grad_params, givens=givens)
    
    #cgh = theano.function(indices+param_tensors, cost_grad_hyper, givens=givens)

    #cgph = theano.function(indices+param_tensors, [jac[0] for jac in cost_grad_params_hyper_by_models], givens=givens)

    vectors = [T.vector() for _ in xrange(k)]
    _new_cgph = T.Lop(cost_grad_params, hyperparams, vectors)
    _new_cgph = theano.function(indices+[beta]+vectors, _new_cgph, updates=model.train_updates,  givens= givens) 

    
    if use_hessian:
        cgpp = T.Lop(cost_grad_params, all_params, vectors)
        cgpp = theano.function(indices+[beta]+vectors, cgpp, updates=model.train_updates,  givens= givens) 
        
    for trial in xrange(trial_num):
        
        
        if verbose>=0:
            print 'trial', trial
        
        for m,w  in zip(training_procedure.models, W0_tensor):
            m.respawn()
            

        #m.params.set_value([ 20.14236541,  99.79383381,  -3.99166711,  -3.05504859])
     
        #print 'preparing', time.time()  - time_s
        time_s = time.time()
        for i in xrange(train_iteration_num):
            if i==reverse_start:
                for m,w  in zip(training_procedure.models, W0_tensor):                
                    w.set_value(m.params.eval())
            res  = training_procedure.do_train()
            
            if verbose>=0 and (verbose==0 or i%verbose==0):
                print 'iteration {0}, internal loss={1} time={2}'.format(str(i), str(res), str(time.time() - time_s))
       		
                time_s = time.time()
	 #print 'training', time.time()  - time_s
        time_s = time.time()
        #W1 = [np.array(m.params.eval()) for m in training_procedure.models]
        
        #if len(history)==0:
        valid_score = training_procedure.do_validation()
        if optimize_learning_rate:                
                history.append(([h.eval() for h in hyperparams]+[learning_rate_param.eval()], valid_score))
        else:
                history.append(([h.eval() for h in hyperparams], valid_score))
        if verbose>=0:
                
            print 'validation score: ', valid_score

        zero_beta = [np.array(0.0).astype(theano.config.floatX)]
        dw = np.array(vgp(*(training_procedure.validation_indices+zero_beta  )))
        #print vgh(*(training_procedure.validation_indices+zero_beta  )  )
        
        dhyp = vgh(*(training_procedure.validation_indices+zero_beta  )  )
        dhyp = np.array([np.array(h)  for h in dhyp]) 
        dv = None
        if optimize_learning_rate:
            dl = 0
        #print  dhyp
        #print 'preparing for reverse', time.time()  - time_s
        time_s = time.time()
        for i in xrange(0,train_iteration_num - reverse_start, reverse_step):
            
            #W = [w0*(i*1.0/train_iteration_num)+w1*((train_iteration_num-i)*1.0/train_iteration_num) for w0, w1 in zip(W0, W1)]
            
            batch = [random.sample(ind, batch_size) for ind in training_procedure.train_indices]        
            #vt = cgp(*(batcn+W))    
            beta_ = [np.array(i*1.0/(train_iteration_num - reverse_start)).astype(theano.config.floatX)]
            
            dv = dw*learning_rate_param.eval()
    
            if use_hessian:
                
                args = batch+beta_+dv.tolist()
                _cgpp = cgpp(*args)


                dw = dw -  np.concatenate(_cgpp).reshape(dw.shape)

            args = batch+beta_+dv.tolist()
            _cgph = (_new_cgph(*args))
            


            _concat_cgph = np.concatenate(_cgph)
            
            dhyp = dhyp - _concat_cgph
            
            if verbose>=0 and (verbose==0 or i%verbose==0):
            
                print 'reverse-iteration {0}, gradients:{1} time:{2}'.format(str(i), str(dhyp), str(time.time() - time_s))
                print np.min((dhyp)), np.max((dhyp))
		time_s = time.time()
		if optimize_learning_rate:
                    print 'lr gradient:', str(dl)
        
        good_update = False 
        attemp_num = 10
        while not good_update and attemp_num>0:
            good_update = True   

            if limits:
                old_values = [h.eval() for h in hyperparams]    
            for h, dh in zip(hyperparams, dhyp):
                #print np.max(abs(h.eval())), np.max(abs(dh))
                h.set_value(np.array(h.eval()) - dh*lr)        
                
            if limits:
                h_id = -1
                for h, l in zip(hyperparams, limits):  
                    h_id+=1 
                    he = h.eval()
                    if np.max((he))>l[1] or np.min((he))<l[0] or np.isnan(np.max(he)) or np.isinf(np.max(he)):
                        print 'bad hyperparam update'
                        print he,' vs limit ',l 
                        if  np.isnan(np.max(he)) or np.isinf(np.max(he)):
                            
                        
                            print 'returning value', o
                            hyperparams[h_id].set_value(old_values[h_id])
                        else:
                            h.set_value(np.minimum(l[1], np.maximum(l[0], he)))
                        lr = lr/10.0
                        print 'new lr', lr
                        attemp_num -= 1
                        good_update = False 
                               


        
        #valid_score = training_procedure.do_validation()
            
        #if verbose>=0:
                
        #print 'validation score: ', valid_score
       
        if optimize_learning_rate:
            good_update = False 
            attemp_num = 10
            while not good_update and attemp_num>0:
                good_update = True   
                if lr_limits:
                    old_values = learning_rate_param.eval()
                learning_rate_param.set_value(learning_rate_param.eval() + lr_for_learning_rate*dl)                
                if lr_limits:
                  
                        le = learning_rate_param.eval()
                        if ((le))>lr_limits[1] or le<lr_limits[0] or np.isnan(le) or np.isinf(le):
                            print 'bad learning rate update'
                            print le,' vs limit ',lr_limits
                            learning_rate_param.set_value(old_values)
                            lr_for_learning_rate = lr_for_learning_rate/10.0
                            print 'new lr', lr
                            attemp_num-=1
                            good_update = False 
                            break       


            
        #    history.append(([h.eval() for h in hyperparams]+[learning_rate_param.eval()], valid_score))
        #else:
        #    history.append(([h.eval() for h in hyperparams], valid_score))
        if verbose>=0 :
                print 'new hyperparam values:', [h.eval() for h in hyperparams]
                #print 'max abs', np.max(abs(np.array([h.eval() for h in hyperparams])))
                if  optimize_learning_rate:
                    print 'new learning rate:', learning_rate_param.eval() 
    
        #print 'reverse', time.time()  - time_s
        time_s = time.time()
    valid_score = training_procedure.do_validation()
    if optimize_learning_rate:                
            history.append(([h.eval() for h in hyperparams]+[learning_rate_param.eval()], valid_score))
    else:
                history.append(([h.eval() for h in hyperparams], valid_score))
                if verbose>=0:

                    print 'validation score: ', valid_score

                
    return HyperparameterOptimization(best_values=history[-1][0], history=history)
        
if __name__=='__main__':
    from generic.optimizer import gd_optimizer
    from models.feedforward import build_feedforward
    from tc.cv import  cv_tc
    from functools import partial

    X, Y = np.load('../../data/W_X_Tr.npy').astype(theano.config.floatX), np.load('../../data/W_Y_Tr.npy').astype(theano.config.floatX)
    X_train = X[:100]
    Y_train = Y[:100]
    X_test = X[100:]
    Y_test = Y[100:]
    print X.shape[1]
    lr = theano.shared(np.array(0.0001).astype(theano.config.floatX))
    alphas = theano.shared(np.array([1.0, 1.0]).astype(theano.config.floatX))
    real_alphas = T.concatenate([T.repeat(alphas[0],  X_train.shape[1] * 50 + 50)   , T.repeat(alphas[1],  50 + 1) ])
    optimizer = partial(gd_optimizer, learning_rate=lr)
    model_build = partial(build_feedforward,  structure = [X_train.shape[1],50, 1],   init_sigmas=[0.001]*3, nonlinearity=lambda x:x, log_alphas =real_alphas, bias=True)


    for _ in xrange(100):
        drmad_optimize(partial(cv_tc,  batch_size=25, k=4),
                    model_build,  
                    optimizer,
                      1, 25,
                     2, 
                    X_train, Y_train, [alphas],  lr,  lr=0.001,  lr_for_learning_rate=0, verbose=1,)
        print alphas.eval()
        print lr.eval()
