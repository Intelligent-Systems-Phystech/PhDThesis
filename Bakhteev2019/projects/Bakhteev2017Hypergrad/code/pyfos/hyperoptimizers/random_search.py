import theano
import theano.tensor as T
import numpy as np
import random 
from structures import HyperparameterOptimization
import time
def random_optimize(trainig_criterion, model_constructor, param_optimizer, trial_num, train_iteration_num, X_data, Y_data, hyperparams, hyperparams_range,  verbose=0):
 
    best = []
    best_score = -np.inf
    history = []
    training_procedure = trainig_criterion( model_constructor, param_optimizer,X_data, Y_data )
        
    for trial in xrange(trial_num):
        for m in training_procedure.models:
            m.respawn()
        
        hyp_value = []
        for r in     hyperparams_range:
            try:
                hyp_value.append(r())
            except:
                hyp_value.append(random.choice(r))

        
        for h, v in zip(hyperparams, hyp_value):
            h.set_value(v)
        if verbose>=0:
            
            print 'hyperparams values:', u' '.join([str(h.eval()) for h in hyperparams])
        time_s = time.time()
        for i in xrange(train_iteration_num):
            res = training_procedure.do_train()
            #print training_procedure.models[0].params.eval()
            if verbose>=0 and (verbose==0 or i%verbose==0):
    
                print 'trial {0} iteration {1}, internal loss={2} time={3}'.format(str(trial), str(i), str(res), str(time.time() - time_s))
		time_s = time.time()
                #print [m.params.eval() for m in training_procedure.models]
    
        score = training_procedure.do_validation()
        history.append((hyp_value[:], score))
        if verbose>=0:
                print 'score {0} vs best {1}'.format(str(score), str(best_score))
        if score>best_score:
            best_score = score
            best = hyp_value[:]
    
    return HyperparameterOptimization(best_values=best, history=history)
        
        
    
    
