from sklearn.datasets import fetch_mldata
import numpy  as np 
#http://g.sweyla.com/blog/2012/mnist-numpy/
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows* cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])#.reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

train_x, train_y = load_mnist(path='/home/legin/mnist')


data_mean = np.mean(train_x, axis=0)
#data_std = np.std(train_x, axis=0)
#data_std[np.where(data_std==0)]=1
train_x = (train_x - data_mean) / 255.0

np.save('mnist_train_x',train_x)
np.save('mnist_train_y',train_y[:,0])
test_x, test_y = load_mnist('testing',path='/home/legin/mnist')



data_mean = np.mean(test_x, axis=0)
data_std = np.std(test_x, axis=0)
#data_std[np.where(data_std==0)]=1
test_x = (test_x - data_mean) / 255.0
np.save('mnist_test_x',test_x)
np.save('mnist_test_y',test_y[:,0])
#test_y.save('test_y')
