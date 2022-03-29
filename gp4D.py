from __future__ import division
import numpy as np
import math 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics
import os
import re
import time 

import open3d as o3d

def expKernel(X, l=1, sig_var=1):
    if isinstance(X, list):
        distance = sklearn.metrics.pairwise_distances(*X)
    else:
        distance = sklearn.metrics.pairwise_distances(X)
    return sig_var*np.exp(-1/(2*l**2)*distance**2)

def thinPlateKernel(X, R=1, sig_var=1):
    if isinstance(X, list):
        distance = sklearn.metrics.pairwise_distances(*X)
    else:
        distance = sklearn.metrics.pairwise_distances(X)
    return 2*np.absolute(distance)**3-3*R*distance**2+R**3 

 
def expGPIS(trainMatX, trainMatY, testMatX, l=1, sig_var=1, noise_var=1):
    Kss = expKernel(testMatX, l, sig_var)
    K = expKernel(trainMatX, l, sig_var)
    L = np.linalg.cholesky(K + noise_var*np.eye(len(trainMatY)))
    K_s = expKernel([trainMatX, testMatX], l, sig_var)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, trainMatY))
    mu = (K_s.T@alpha).squeeze()
    v = np.linalg.solve(L, K_s)
    cov =  Kss - v.T@v
    stdv = np.sqrt(np.diag(cov)).squeeze()
    return mu, cov

def thinGPIS(trainMatX, trainMatY, testMatX, R=1, sig_var=1, noise_var=1):
    Kss = thinPlateKernel(testMatX, R, sig_var)
    K = thinPlateKernel(trainMatX, R, sig_var)
    L = np.linalg.cholesky(K + noise_var*np.eye(len(trainMatY)))
    K_s = thinPlateKernel([trainMatX, testMatX], R, sig_var)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, trainMatY))
    mu = (K_s.T@alpha).squeeze()
    v = np.linalg.solve(L, K_s)
    cov =  Kss - v.T@v
    stdv = np.sqrt(np.diag(cov)).squeeze()
    return mu, cov

#with open('/home/user/GPIS/data/pitcherC.off') as file:
#    points = file.readlines()
#    pointN = len(points)-2
#    pcdPoints = np.zeros((pointN,3))
#    for i,point in enumerate(points):
#        if i > 1:
#            values = point.split()
#            point = np.array([float(values[0]),float(values[1]),float(values[2])])
#            pcdPoints[i-2,:] = point

pcd = o3d.io.read_point_cloud("../3d-tools/python/mesh_pc_render/partial_pc/019_pitcher_base_0000_pc.pcd")
pcdPoints = np.asarray(pcd.points)
print(np.asarray(pcd.points))
o3d.visualization.draw_geometries([pcd])

print("Compute the normal of the downsampled point cloud")
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd)
o3d.visualization.draw_geometries([pcd],point_show_normal=True)

pointN = 3000
trainN = 2000
#mask = np.zeros(pointN, dtype=bool)
#mask[np.random.choice(np.arange(0,pointN), size=trainN, replace=False)] = True

mask = np.zeros(pointN, dtype=bool)
mask[np.random.choice(np.arange(0,pointN), size=trainN, replace=False)] = True

trainMatX = np.column_stack([pcdPoints[mask,0],pcdPoints[mask,1],pcdPoints[mask,2]])
trainMatY = np.zeros((trainN,1))

trainMatXOutside = np.zeros((trainN,3))
trainMatYOutside = np.ones((trainN,1))

trainMatXInside = np.zeros((trainN,3))
trainMatYInside = (-1)*np.ones((trainN,1))

import ipdb; ipdb.set_trace()

# Add point inside and outside 
outDim = 0.001
pcdNormals = np.asarray(pcd.normals)
for index,normal in enumerate(pcdNormals[mask]):
    xOut = trainMatX[index,0] + outDim*normal[0] #np.dot(normal,np.array([0,0,1]))
    yOut = trainMatX[index,1] + outDim*normal[1] 
    zOut = trainMatX[index,2] + outDim*normal[2] 
    trainMatXOutside[index,:] = np.array([xOut,yOut,zOut])
    xIn = trainMatX[index,0] + -outDim*normal[0]
    yIn = trainMatX[index,1] + -outDim*normal[1]
    zIn = trainMatX[index,2] + -outDim*normal[2]
    trainMatXInside[index,:] = np.array([xIn,yIn,zIn])

trainMatXAll = np.row_stack((trainMatX,trainMatXOutside,trainMatXInside))
trainMatYAll = np.row_stack((trainMatY,trainMatYOutside,trainMatYInside))

print(trainMatXAll.shape)
print(trainMatYAll.shape)
    
xV = np.linspace(-0.1,0.1,30)
yV = np.linspace(-0.1,0.1,30)
zV = np.linspace(-0.12,0.12,30)

#x, y, z = np.meshgrid(xV, yV, zV)
#testMatX = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
testMatX = np.column_stack([pcdPoints[~mask,0],pcdPoints[~mask,1],pcdPoints[~mask,2]])

## GPIS ##
R = 1.7320*0.3
varS = 0.0001
varN = 0.0001

#startTime = time.time()
#mu, cov = expGPIS(trainMatXAll, trainMatYAll, testMatX, l, varS, varN)
#print("--- Exp %s seconds ---" % (time.time() - startTime))

startTime = time.time()
mu, cov = thinGPIS(trainMatXAll, trainMatYAll, testMatX, R, varS, varN)
print("--- Thin %s seconds ---" % (time.time() - startTime))

print(np.mean(mu))
std=np.sqrt(np.diag(cov)).squeeze()
figC = plt.figure()
plt.plot(mu, color='b')
plt.plot(mu+std, color='r')
plt.plot(mu-std, color='r')
plt.show() 

fig3D = plt.figure(figsize=plt.figaspect(1))  
ax = fig3D.gca(projection='3d')
ax.scatter(trainMatX[:,0], trainMatX[:,1], trainMatX[:,2], color='g')
ax.scatter(trainMatXOutside[:,0], trainMatXOutside[:,1], trainMatXOutside[:,2], color='r')
ax.scatter(trainMatXInside[:,0], trainMatXInside[:,1], trainMatXInside[:,2], color='b')
plt.show()

std = np.sqrt(np.diag(cov)).squeeze()
indexes = np.absolute(mu)<0.02

fig3D = plt.figure(figsize=plt.figaspect(1))  
ax = fig3D.gca(projection='3d')
ax.scatter(trainMatX[:,0], trainMatX[:,1], trainMatX[:,2], color='g')
ax.scatter(testMatX[indexes,0], testMatX[indexes,1], testMatX[indexes,2], c=std[indexes] , cmap='hot')
plt.show()












 