import numpy as np
import rbm
import ae
import softmax
import pickle
import theano
import sys

args = sys.argv

theano.config.openmp = True
"""
syntax:
python launch.py WISDM_filename RBM_layer_size AE_layer_size Softmax_layer_size RBM_epochs AE_epochs Softmax_epochs part_of_data_for_training number_of_iterations 

Note, that test dataset always contains 25% of the original dataset
example:
python launch.py ../data/WISDM.npy 378 225 117 500 500 3000 0.75 5
"""
try:
    fname = args[1]
    rbm_size = int(args[2])
    ae_size = int(args[3])
    sm_size = int(args[4])
    rbm_epochs = int(args[5])
    ae_epochs = int(args[6])
    sm_epochs = int(args[7])
    train_part = float(args[8])
    iters = int(args[9])
    if train_part > 0.75:
        print 'warning: train and test part will intersect during cross validation'

except:
    print 'syntax incorrect'
    print 'python launch.py WISDM_filename RBM_layer_size AE_layer_size Softmax_layer_size RBM_epochs AE_epochs Softmax_epochs part_of_data_for_training number_of_iterations '
    exit(1)

data0 = np.load(fname)


def do_launch():
    """
    one cross-validation launch
    """
    data = np.copy(data0)
    datas = []
    # balancing data
    for c in range(0, 6):
        datas.append(np.repeat(np.where(data[:, 0] == c), 20))
    for c in range(1, 6):
        delta = sum(data[:, 0] == 0) - sum(data[:, 0] == c)
        if delta > 0:
            ready = False
        np.random.shuffle(datas[c])
        data = np.concatenate((data, data[datas[c][:delta], :]))
    data_mean = np.mean(data[:, 1:], axis=0)
    data_std = np.std(data[:, 1:], axis=0)
    # standartize
    data[:, 1:] = (data[:, 1:] - data_mean) / data_std
    np.random.shuffle(data)
    print 'train size:' + str(int(data.shape[0] * train_part))
    train = data[0:int(data.shape[0] * train_part), 1:]
    train_y = data[0:int(data.shape[0] * train_part), 0]
    rbm.train_rbm(dataset=train, training_epochs=rbm_epochs, batch_size=20, n_hidden=rbm_size)
    ae.train_ae(dataset=train, training_epochs=ae_epochs, batch_size=20, n_visible=rbm_size, n_hidden=ae_size)
    softmax.train_softmax(datasets=(train, train_y.astype(int)), learning_rate=0.25, n_epochs=sm_epochs, n_in=ae_size,
                          n_hid=sm_size, start_num=8)
    # loading model after softmax training
    with open('best_model', 'rb') as inp:
        classifier = pickle.load(inp)
    test = data[int(data.shape[0] * 0.75):, 1:]
    test_y = data[int(data.shape[0] * 0.75):, 0]
    new_y = classifier.predict_model(test)
    new_y_train = classifier.predict_model(train)
    err = 0
    # Note that the training errors are computed before the weights update, so thefinal error on train
    # can be slightly different from the last error in train_softmax function
    err = np.mean(np.not_equal(new_y, test_y))
    err_train = np.mean(np.not_equal(new_y_train, train_y))
    return err, err_train


ers = []
ers2 = []
for i in range(0, iters):
    er, er2 = do_launch()
    print er
    print er2
    ers.append(er)
    ers2.append(er2)
print 'test errors (mean and std):'
print ers, np.mean(ers), np.std(ers)
print 'train errors (mean and std):'
print ers2, np.mean(ers2), np.std(ers2)
