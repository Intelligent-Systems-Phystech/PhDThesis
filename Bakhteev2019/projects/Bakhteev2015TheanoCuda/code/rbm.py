"""
This source was modified from tutorial at http://deeplearning.net/tutorial/
for deal with real values in features.
The train algorithm was  inspired by matlab implemintation of RBM:
http://www.mathworks.com/matlabcentral/fileexchange/42853-deep-neural-network
I've found it more efficient than versions that were discussed on the Theano forums
"""
import timeit
import numpy
import theano
import theano.tensor as T
import os
from theano.tensor.shared_randomstreams import RandomStreams
from utils import to_shared_data


class RBM(object):
    """Restricted Boltzmann Machine (RBM)  """

    def __init__(
            self,
            input=None,
            n_visible=600,
            n_hidden=200,
            W=None,
            hbias=None,
            vbias=None,
    ):

        self.n_visible = n_visible
        self.n_hidden = n_hidden


        # create a number generator
        numpy_rng = numpy.random.RandomState(None)
        theano_rng = RandomStreams(None)
        if W is None:
            # W is initialized with `initial_W` which is uniformely
            # sampled from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible)) the output of uniform if
            # converted using asarray to dtype theano.config.floatX so
            # that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            # theano shared variables for weights and biases
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if hbias is None:
            # create shared variable for hidden units bias
            hbias = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='hbias',
                borrow=True
            )

        if vbias is None:
            # create shared variable for visible units bias
            vbias = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                name='vbias',
                borrow=True
            )

        # initialize input layer for standalone RBM or layer0 of DBN
        self.input = input
        if not input:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng
        # **** WARNING: It is not a good idea to put things in this list
        # other than shared variables created in this function.
        self.params = [self.W, self.hbias, self.vbias]

    def propup(self, vis):
        '''This function propagates the visible units activation upwards to
        the hidden units

        Note that we return also the pre-sigmoid activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        ''' This function infers state of hidden units given visible units '''
        # compute the activation of the hidden units given a sample of
        # the visibles
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of the hiddens given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        h1_sample = self.theano_rng.binomial(size=h1_mean.shape,
                                             n=1, p=h1_mean,
                                             dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        '''This function propagates the hidden units activation downwards to
        the visible units

        Note that we return also the pre_sigmoid_activation of the
        layer. As it will turn out later, due to how Theano deals with
        optimizations, this symbolic variable will be needed to write
        down a more stable computational graph (see details in the
        reconstruction cost function)

        '''
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        mean = T.nnet.sigmoid(pre_sigmoid_activation)
        return [pre_sigmoid_activation, mean]

    def sample_v_given_h(self, h0_sample):
        ''' This function infers state of visible units given hidden units '''
        # compute the activation of the visible given the hidden sample
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        # get a sample of the visible given their activation
        # Note that theano_rng.binomial returns a symbolic sample of dtype
        # int64 by default. If we want to keep our computations in floatX
        # for the GPU we need to specify to return the dtype floatX
        # v1_sample = self.theano_rng.binomial(size=v1_mean.shape,
        #                                     n=1, p=v1_mean,
        #                                     dtype=theano.config.floatX)
        # v1_sample =  pre_sigmoid_v1 + self.theano_rng.normal(size=v1_mean.shape, avg=0.0, std=1.0, dtype=theano.config.floatX)
        v1_sample = pre_sigmoid_v1
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1):

        # compute positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        vis0 = self.input

        hid0 = self.sample_h_given_v(vis0)[1]
        hid1 = hid0
        bhid0 = self.theano_rng.uniform(size=hid0.shape)
        bhid0 = 1 * (bhid0 < hid0)
        vis1 = self.sample_v_given_h(bhid0)[0]
        hid1 = self.sample_h_given_v(vis1)[1]

        posprods = T.dot(hid0.T, vis0)
        negprods = T.dot(hid1.T, vis1)
        dW = (posprods - negprods).T;
        dB = (T.sum(hid0, axis=0) - T.sum(hid1, axis=0))
        dC = (T.sum(vis0, axis=0) - T.sum(vis1, axis=0))

        from  collections import OrderedDict
        updates = OrderedDict()
        updates[self.W] = self.W + dW * T.cast(
            lr,
            dtype=theano.config.floatX)
        updates[self.vbias] = self.vbias + dC * T.cast(
            lr,
            dtype=theano.config.floatX)
        updates[self.hbias] = self.hbias + dB * T.cast(
            lr,
            dtype=theano.config.floatX)
        err = (vis0 - vis1) ** 2
        rmse = T.mean(err)
        return rmse, updates


def train_rbm(learning_rate=0.00001, training_epochs=10,
              dataset=None, batch_size=100,
              n_hidden=300, n_visible=600):
    """
    Demonstrate how to train RBM. After that pickles model as file 'rbm'

    """
    train_set_x = to_shared_data(dataset)
    # compute number of minibatches for training, validation and testing
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = numpy.random.RandomState(None)
    theano_rng = RandomStreams(None)

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    # construct the RBM class
    rbm = RBM(input=x, n_visible=n_visible,
              n_hidden=n_hidden)

    # get the cost and the gradient corresponding to one step of CD-1
    cost, updates = rbm.get_cost_updates(lr=learning_rate)

    #################################
    #     Training the RBM          #
    #################################


    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in range(training_epochs):
        mean_cost = []
        for batch_index in range(n_train_batches):
            if batch_index % 100 == 0:
                mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time)

    print 'Training took %f minutes' % (pretraining_time / 60.)
    import pickle
    with open('rbm', 'wb') as out:
        pickle.dump(rbm, out)
