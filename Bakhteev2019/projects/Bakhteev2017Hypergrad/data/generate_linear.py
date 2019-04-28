import numpy as np

sz = 40
rs = np.random.RandomState(42)
X = rs.randn(sz)
XX = np.vstack([ X,np.ones(sz), X**2, X**3, X**4, X**5, X**6, X**7, X**8, X**9,  np.cos(X), np.sin(X)])
XX = XX.T

bias =  np.mean(XX, axis=0)
var = (np.std(XX, axis=0))
XX  = (XX-bias)/var # division by zero in second column, it's OK
XX[:,1] =  np.ones(sz)
dim = XX.shape[1]

Y = XX[:,0]  + rs.randn(sz)*1.0

np.save('linear2_x', XX)
np.save('linear2_y', Y)

