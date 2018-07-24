import numpy
import theano
import theano.tensor as T
import numpy as np
import random
from structures import TrainingProcedure

def simple_tc(model_constructor, param_optimizer,X_data, Y_data, batch_size=1):
    X = T.matrix()
    if len(Y_data.shape)==1:
        Y = T.vector(dtype=Y_data.dtype)   
    else:
        Y = T.matrix()
    data_size = X_data.shape[0]
    X_data = theano.shared(X_data)
    Y_data = theano.shared(Y_data)
    index = T.ivector()
    model = model_constructor( dataset_size=data_size)
    cost = model.cost(X,Y)
    validation = model.validation(X, Y)

    updates = param_optimizer( cost, model.params)
    updates.extend(model.train_updates)
    train_func = theano.function([index], cost,  updates= updates, givens= [(X, X_data[index]), (Y, Y_data[index])])
    monitor = theano.function([index], validation,   givens= [(X, X_data[index]), (Y, Y_data[index])])
    #grads = theano.function([index],  T.grad(cost, model.params),  givens= [(X, X_data[index]), (Y, Y_data[index])])

    def do_train(callback=None):
        
        indexes = random.sample(range(data_size), batch_size)
        
        if callback:
            cres = callback([indexes])
        
        res = train_func(indexes)
        if callback:
           
            return cres, res
        #print grads(indexes)

        return res
    
    def do_validation():
        return  monitor(range(data_size))
    procedure = TrainingProcedure(do_train = do_train, do_validation = do_validation, X_tensors = [X], Y_tensors = [Y],  models=[model], updates = updates,
    train_indices=[range(data_size)], validation_indices=[range(data_size)])
    return procedure
    
        
    
