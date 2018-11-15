import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
import ae
import pickle
from utils import to_shared_data


class SM(object):
    """Multi-class Logistic Regression Class (2-layer net)

    This source was modified from tutorial at http://deeplearning.net/tutorial/
    This implementation works only in stack with pickled model of autoencoder
    """

    def __init__(self, input, n_in, n_out, n_out2, ae_path='ae'):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of units in the first layer

        :param n_out2: number of units the second layer(output)

       
        :param ae_path: path to the pickled autoencoder 

        """
        numpy_rng = numpy.random.RandomState(None)
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_in + n_out)),
                    high=4 * numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                # numpy.zeros((n_in,n_out)),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.W2 = theano.shared(
            value=numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_out + n_out2)),
                    high=4 * numpy.sqrt(6. / (n_out + n_out2)),
                    size=(n_out, n_out2)
                ),
                # numpy.zeros((n_out,n_out2)),
                dtype=theano.config.floatX
            ),
            name='W2',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b2 = theano.shared(
            value=numpy.zeros(
                (n_out2,),
                dtype=theano.config.floatX
            ),
            name='b2',
            borrow=True
        )

        with open(ae_path, 'rb') as inp:
            self.ae = pickle.load(inp)
        # build layers
        first_layer = T.dot(self.input_to_features(input), self.W) + self.b
        second_layer = T.dot(T.tanh(first_layer), self.W2) + self.b2
        self.p_y_given_x = T.nnet.softmax(second_layer)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model, we train the model by fine-tuning method, i.e.
        # we use not only params of this net, but also the params of the previous layer
        self.params = [self.W, self.b, self.ae.rbm.W, self.ae.rbm.hbias, self.ae.rbm.vbias,
                       self.ae.W, self.ae.b, self.ae.b_prime]

        # keep track of model input        
        self.input = input

        self.predict_model = theano.function(
            inputs=[self.input],
            outputs=self.y_pred, allow_input_downcast=True)

    def input_to_features(self, inp):
        [_, rbm_out, _] = self.ae.rbm.sample_h_given_v(inp)
        z = self.ae.get_hidden_values(rbm_out)
        return z

    def monitor_cost(self, y):
        """Return the mean probability of the real labels 
           this value can be used as a monitor value during training
         """

        return  -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
#-T.mean((self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


def train_softmax(learning_rate=0.2, n_epochs=20000, n_in=200, n_hid=100,
                  datasets=None,
                  batch_size=None, start_num=4):
    """
    trains net using multistart with start_num starts, selects the best and pickles it
    """
    if batch_size == None:
        batch_size = int(datasets[0].shape[0])

    train_set_x = to_shared_data(datasets[0])
    train_set_y = to_shared_data(datasets[1], dtype='int32')
    # compute number of minibatches for training, validation and testing
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################


    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels
    classifiers = []
    outputs_ = []
    updates_ = []
    for i in range(0, start_num):
        classifier = SM(input=x, n_in=n_in, n_out=n_hid, n_out2=6)

        # the cost we minimize during training is the negative log likelihood of
        # the model in symbolic format
        cost = classifier.monitor_cost(y)


        # compute the gradient of cost with respect to theta = (W,b)
        g_W = T.grad(cost=cost, wrt=classifier.W)
        g_b = T.grad(cost=cost, wrt=classifier.b)
        g_W2 = T.grad(cost=cost, wrt=classifier.W2)
        g_b2 = T.grad(cost=cost, wrt=classifier.b2)
        g_rbW = T.grad(cost=cost, wrt=classifier.ae.rbm.W)
        g_rbh = T.grad(cost=cost, wrt=classifier.ae.rbm.hbias)
        g_aeW = T.grad(cost=cost, wrt=classifier.ae.W)
        g_aeb = T.grad(cost=cost, wrt=classifier.ae.b)



        # start-snippet-3
        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs.
        updates = [(classifier.W, classifier.W - learning_rate * g_W),
                   (classifier.b, classifier.b - learning_rate * g_b),
                   (classifier.ae.rbm.W, classifier.ae.rbm.W - learning_rate * g_rbW),
                   (classifier.ae.rbm.hbias, classifier.ae.rbm.hbias - learning_rate * g_rbh),
                   (classifier.ae.W, classifier.ae.W - learning_rate * g_aeW),
                   (classifier.ae.b, classifier.ae.b - learning_rate * g_aeb),
                   (classifier.W2, classifier.W2 - learning_rate * g_W2),
                   (classifier.b2, classifier.b2 - learning_rate * g_b2)
                   ]
        # updates_ and outputs_ are used for multistart
        updates_.extend(updates)
        outputs_.append(classifier.errors(y))
        classifiers.append((classifier))

    train_model = theano.function(
        inputs=[index],
        outputs=outputs_,
        updates=updates_,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    ###############
    # TRAIN MODEL #
    ###############



    start_time = timeit.default_timer()
    epoch = 0

    while (epoch < n_epochs):
        epoch = epoch + 1
        minibatch_avg_cost = []
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost.append(train_model(minibatch_index))
            print numpy.mean(minibatch_avg_cost, axis=0) * 100,'epoch ',epoch

    print  'best is ', numpy.argmin(numpy.mean(minibatch_avg_cost, axis=0))
    best = classifiers[numpy.argmin(minibatch_avg_cost)]


    # save the best model
    with open('best_model', 'wb') as f:
        pickle.dump(best, f)
    end_time = timeit.default_timer()
