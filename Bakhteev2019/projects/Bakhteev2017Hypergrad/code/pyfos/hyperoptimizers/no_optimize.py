import theano
import theano.tensor as T
import numpy as np
import random 
from structures import HyperparameterOptimization
def no_optimize(trainig_criterion, model_constructor, param_optimizer,  train_iteration_num, X_data, Y_data, verbose=0):
 
        
    training_procedure = trainig_criterion( model_constructor, param_optimizer,X_data, Y_data )
    
    for i in xrange(train_iteration_num):

        res = training_procedure.do_train()
       
                
        
       
    return HyperparameterOptimization()
        
        
    
    
