import theano
import theano.tensor as T
def none_optimizer(cost, params):
    return [(p,p) for p in params]

def gd_optimizer(cost, params, learning_rate=1.0):
    if not isinstance(params, list):
        params = [params]

    grad  = T.grad(cost, params)

    return [(p, p + g*learning_rate) for g, p in zip(grad, params)]
