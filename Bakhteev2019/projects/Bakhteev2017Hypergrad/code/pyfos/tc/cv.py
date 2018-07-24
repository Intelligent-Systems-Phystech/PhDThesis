import numpy
import theano
import theano.tensor as T
import numpy as np
import random
from structures import TrainingProcedure
import time 
def cv_tc(model_constructor, param_optimizer,X_data, Y_data, k = 1, validation_part=0.25,   batch_size=1):
    Xs = [T.matrix() for _ in xrange(k)]
    Ys = [T.vector(dtype=Y_data.dtype) for _ in xrange(k)]
    indices = [T.ivector() for _ in xrange(k)]

    data_size = X_data.shape[0]
    X_data = theano.shared(X_data)
    Y_data = theano.shared(Y_data)
    
    updates = []
    givens = []
    costs = []
    models = []
    validations = []
    for X,Y, index in zip(Xs, Ys, indices):
    
        model = model_constructor(dataset_size=int(data_size*(1-validation_part)))
        cost = model.cost(X,Y)
        valid = model.validation(X,Y)

        updates.extend(param_optimizer( cost, model.params))
        updates.extend(model.train_updates)
        givens.append((X,X_data[index]))
        givens.append((Y,Y_data[index]))
        costs.append(cost)
        models.append(model)
        validations.append(valid)

    train_func = theano.function(indices, T.mean(costs),  updates= updates, givens= givens)
    monitor = theano.function(indices, T.mean(validations),  givens= givens)
    
    train_indices, valid_indices =[],[]
    for _ in xrange(k):
        ind = range(data_size)
        random.shuffle(ind)
        train_indices.append(ind[:int(len(ind)*(1-validation_part))])
        valid_indices.append(ind[int(len(ind)*(1-validation_part)):])


    def do_train(callback=None):
        time_s = time.time()
        indices = []
        for k_ in xrange(k):

            sub_indices  = random.sample(train_indices[k_], batch_size)
            indices.append(sub_indices)
        if callback:
            call_res = callback(indices)
        else:
            call_res = None
        #print 'ind', time_s - time.time()
        res = train_func(*indices)
        #print 'train', time_s - time.time()
        if call_res:
            return res, call_res
        return res 
    
    def do_validation():
        indices = []
        for k_ in xrange(k):
            sub_indices  = valid_indices[k_]
            indices.append(sub_indices)
       
        return  monitor(*indices)
    procedure=TrainingProcedure(do_train = do_train, do_validation = do_validation, X_tensors = Xs, Y_tensors = Ys, models=models, updates = updates,
    train_indices=train_indices, validation_indices=valid_indices)
    return procedure
    
        
    
