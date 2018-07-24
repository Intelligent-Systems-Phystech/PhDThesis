class Model:
    def __init__(self, cost = None, validation = None, params=None, predict = None, respawn=None, train_updates=[]):
        self.cost = cost
        self.params = params
        self.predict_var = predict
        self.respawn = respawn
        self.validation = validation
        self.train_updates = train_updates

class HyperparameterOptimization:
    def __init__(self, best_values=None, history=None):
        self.history= history
        self.best_values = best_values


class TrainingProcedure:
    def __init__(self, do_train = None, do_validation = None, X_tensors = None, Y_tensors = None, models=None,  updates=None, train_indices=None, validation_indices=None):
        self.do_train = do_train
        self.do_validation = do_validation
        self.X_tensors = X_tensors
        self.Y_tensors = Y_tensors       
        self.models = models
        self.updates = updates
        self.train_indices = train_indices
        self.validation_indices = validation_indices
        



