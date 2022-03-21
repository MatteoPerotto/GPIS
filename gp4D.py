from __future__ import division
import numpy as np
import math 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics
import os
import re
import time 



def kernel(X, l=1, sig_var=1):
    if isinstance(X, list):
        distance = sklearn.metrics.pairwise_distances(*X)
    else:
        distance = sklearn.metrics.pairwise_distances(X)
    return sig_var*np.exp(-1/(2*l**2)*distance**2)


def GPIS(trainMatX, trainMatY, testMatX, l=1, sig_var=1, noise_var=1):
    Kss = kernel(testMatX, l, sig_var)
    K = kernel(trainMatX, l, sig_var)
    L = np.linalg.cholesky(K + noise_var*np.eye(len(trainMatY)))
    K_s = kernel([trainMatX, testMatX], l, sig_var)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, trainMatY))
    mu = (K_s.T@alpha).squeeze()
    v = np.linalg.solve(L, K_s)
    cov =  Kss - v.T@v
    stdv = np.sqrt(np.diag(cov)).squeeze()
    return mu, cov


with open('/home/user/GPIS/data/pitcherC.off') as file:
    points = file.readlines()
    pointN = len(points)-2
    meshPoints = np.zeros((pointN,3))
    for i,point in enumerate(points):
        if i > 1:
            values = point.split()
            point = np.array([float(values[0]),float(values[1]),float(values[2])])
            meshPoints[i-2,:] = point


trainN = 1000
mask = np.zeros(pointN, dtype=bool)
mask[np.random.choice(np.arange(0,pointN), size=trainN, replace=False)] = True

trainMatX = np.column_stack([meshPoints[mask,0],meshPoints[mask,1],meshPoints[mask,2]])
trainMatY = np.zeros(trainN)

#testMatX = np.column_stack([meshPoints[~mask,0],meshPoints[~mask,1],meshPoints[~mask,2]])

xV = np.linspace(-0.1,0.1,10)
yV = np.linspace(-0.1,0.1,10)
zV = np.linspace(-0.15,0.15,10)

x, y, z = np.meshgrid(xV, yV, zV)
testMatX = np.column_stack([x.ravel(), y.ravel(), z.ravel()])


#import ipdb; ipdb.set_trace()
## GPIS ##
l = 1.4
varS = 0.05
varN = 0.1

startTime = time.time()
mu, cov = GPIS(trainMatX, trainMatY, testMatX, l, varS, varN)
print("--- %s seconds ---" % (time.time() - startTime))


## #### ##

print(np.mean(mu))
std=np.sqrt(np.diag(cov)).squeeze()
figC = plt.figure()
plt.plot(mu, color='b')
plt.plot(mu+std, color='r')
plt.plot(mu-std, color='r')
plt.show()

fig3D = plt.figure(figsize=plt.figaspect(1))  
ax = fig3D.gca(projection='3d')
ax.scatter(meshPoints[mask,0], meshPoints[mask,1], meshPoints[mask,2], color='g')
#ax.scatter(meshPoints[~mask,0], meshPoints[~mask,1], meshPoints[~mask,2], color='r')
ax.scatter(x.ravel(), y.ravel(), z.ravel(), c=std , cmap='plasma')
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-0.15, 0.15))
plt.show()

maskGrid = np.zeros(len(mu), dtype=bool)
maskGrid[np.abs(std)<0.015] = True 

fig3D = plt.figure(figsize=plt.figaspect(1))  
ax = fig3D.gca(projection='3d')
ax.scatter(meshPoints[mask,0], meshPoints[mask,1], meshPoints[mask,2], color='g')
ax.scatter(x.ravel()[maskGrid], y.ravel()[maskGrid], z.ravel()[maskGrid], c=std[maskGrid] , cmap='plasma')
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-0.15, 0.15))
plt.show()













 