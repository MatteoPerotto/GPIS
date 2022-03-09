from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics


dlim = 5
nTrX, nTrY = (50, 50)
nTsX, nTsY = (20, 20)
l = 3
varS = 0.1
varN = 4

xTst, yTst = np.meshgrid(np.linspace(-dlim, dlim, nTsX),np.linspace(-dlim, dlim, nTsY))
x = xTst.ravel()
y = yTst.ravel()
testX = np.column_stack([x,y])
print("Test: ", testX.shape)

xTr, yTr = np.meshgrid(np.linspace(-dlim, dlim, nTrX),np.linspace(-dlim, dlim, nTrY))
x = xTr.ravel()
y = yTr.ravel()
trainX = np.column_stack([x,y])
trainY = np.sin(np.sqrt(x**2+y**2))
print("trainX:", trainX.shape)
print("trainY:", trainY.shape)


# Define the kernel for R2 points
def kernel(X, l=1, sig_var=1):
    if isinstance(X, list):
        distance = sklearn.metrics.pairwise_distances(*X)
    else:
        distance = sklearn.metrics.pairwise_distances(X)
    return sig_var*np.exp(-1/(2*l**2)*distance**2)

def kernelP(X, l=1, sig_var=1):
    p = 2
    if isinstance(X, list):
        distance = sklearn.metrics.pairwise_distances(*X)
    else:
        distance = sklearn.metrics.pairwise_distances(X)
    return sig_var*np.exp(-(2*np.sin(np.pi*distance/p)**2)/(l**2))


def GP3D(trainX, trainY, testX, l=1, sig_var=1, noise_var=1):
    Kss = kernel(testX, l, sig_var)
    K = kernel(trainX, l, sig_var)
    L = np.linalg.cholesky(K + noise_var*np.eye(len(trainY)))
    K_s = kernel([trainX, testX], l, sig_var)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, trainY))
    mu = (K_s.T@alpha).squeeze()
    v = np.linalg.solve(L, K_s)
    cov =  Kss - v.T@v
    #stdv = np.sqrt(np.diag(cov)).squeeze()
    return mu, cov

mu, cov = GP3D(trainX, trainY, testX, l, varS, varN)


fig3D = plt.figure(figsize=plt.figaspect(1))  # Square figure
ax = fig3D.gca(projection='3d')
ax.scatter(xTr, yTr, trainY.reshape(nTrX,nTrY), color='b')
ax.scatter(xTst, yTst, mu.reshape(nTsX,nTsY), color='r')
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-dlim, dlim))

getattr(ax, 'set_{}lim'.format('z'))((-1, 1))
plt.show()
fig3D.savefig('3D.png')


halfTrain = int(nTrX*nTrX/2)
halfTest = int(nTsX*nTsX/2)
fig2D = plt.figure(figsize=plt.figaspect(1))  # Square figure
plt.scatter(xTst.ravel()[halfTest:halfTest+nTsX], mu[halfTest:halfTest+nTsX], color='b')
plt.scatter(trainX[halfTrain:halfTrain+nTrX,0], trainY[halfTrain:halfTrain+nTrX], color='r')
plt.show()
fig2D.savefig('2D.png')



 