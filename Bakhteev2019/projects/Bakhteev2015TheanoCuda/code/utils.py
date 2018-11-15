import numpy
import theano


def to_shared_data(dataset, dtype=theano.config.floatX):
    """
    this code is to transfer the dataset into theano shared data, that is usefull 
    when running code on gpu
    """

    shared = theano.shared(numpy.asarray(dataset, dtype=dtype),
                           borrow=True)

    return shared
