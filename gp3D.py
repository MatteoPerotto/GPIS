from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics


dlim = 5
nTrainX, nTrainY = (50, 50)
nTestX, nTestY = (20, 20)
l = 1
varS = 0.1
varN = 4

xTest, yTest = np.meshgrid(np.linspace(-dlim, dlim, nTestX),np.linspace(-dlim, dlim, nTestY))
x = xTest.ravel()
y = yTest.ravel()
testMatX = np.column_stack([x,y])
print("Test: ", testMatX.shape)

xTrain, yTrain = np.meshgrid(np.linspace(-dlim, dlim, nTrainX),np.linspace(-dlim, dlim, nTrainY))
x = xTrain.ravel()
y = yTrain.ravel()
trainMatX = np.column_stack([x,y])
trainMatY = np.sin(np.sqrt(x**2+y**2))
print("trainMatX:", trainMatX.shape)
print("trainMatY:", trainMatY.shape)


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


def GP3D(trainMatX, trainMatY, testMatX, l=1, sig_var=1, noise_var=1):
    Kss = kernel(testMatX, l, sig_var)
    K = kernel(trainMatX, l, sig_var)
    L = np.linalg.cholesky(K + noise_var*np.eye(len(trainMatY)))
    K_s = kernel([trainMatX, testMatX], l, sig_var)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, trainMatY))
    mu = (K_s.T@alpha).squeeze()
    v = np.linalg.solve(L, K_s)
    cov =  Kss - v.T@v
    #stdv = np.sqrt(np.diag(cov)).squeeze()
    return mu, cov

mu, cov = GP3D(trainMatX, trainMatY, testMatX, l, varS, varN)


fig3D = plt.figure(figsize=plt.figaspect(1))  # Square figure
ax = fig3D.gca(projection='3d')
ax.scatter(xTrain, yTrain, trainMatY.reshape(nTrainX,nTrainY), color='r')
ax.scatter(xTest, yTest, mu.reshape(nTestX,nTestY), color='b')
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-dlim, dlim))

getattr(ax, 'set_{}lim'.format('z'))((-1, 1))
plt.show()
fig3D.savefig('3D.png')


halfTrain = int(nTrainX*nTrainX/2)
halfTest = int(nTestX*nTestX/2)
fig2D = plt.figure(figsize=plt.figaspect(1))  # Square figure
plt.scatter(xTest.ravel()[halfTest:halfTest+nTestX], mu[halfTest:halfTest+nTestX], color='b')
plt.scatter(trainMatX[halfTrain:halfTrain+nTrainX,0], trainMatY[halfTrain:halfTrain+nTrainX], color='r')
plt.show()
fig2D.savefig('2D.png')



 