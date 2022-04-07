from __future__ import division
import numpy as np
import math 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import time 
import open3d as o3d

import torch 
import gpytorch 

import gpdata 
import myGPs

# Import point cloud
pcd = o3d.io.read_point_cloud("../3d-tools/python/mesh_pc_render/partial_pc/002_master_chef_can_0000_pc.pcd")
pcdPoints = np.asarray(pcd.points)
#o3d.visualization.draw_geometries([pcd])

# Center it  
pcdCenter = pcd.get_center()

# Estimate normals 
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd)
pcdNormals = np.asarray(pcd.normals)
#o3d.visualization.draw_geometries([pcd],point_show_normal=True)

# Estimate bounding box 
boundingBox = o3d.geometry.PointCloud.get_oriented_bounding_box(pcd)
boundingBox.color = np.asarray([0, 0, 0])
boundingBox.scale(1.5,boundingBox.center)
o3d.visualization.draw_geometries([pcd,boundingBox])

boxCenter = boundingBox.center
boxDim = boundingBox.extent
boxPoints = np.asarray(boundingBox.get_box_points())
maxDiag = np.linalg.norm(boxDim)
            
trainN = 1000
dataHandler = gpdata.GPdataHandler(pcdPoints, pcdCenter, trainN)
#trainMatXAll, trainMatYAll = dataHandler.genFromNormals(pcdNormals, 0.01)
trainMatXAll, trainMatYAll = dataHandler.genFromBB(boxCenter, boxPoints, boxDim, boxEdgePointN=5, spherePointN=3,sphereRadius=0.01)

testPointN = 20000
testMatX = dataHandler.genTestPoints(testPointN,boxCenter,boxDim)

Xt = torch.from_numpy(trainMatXAll).float()
Yt = torch.from_numpy(np.squeeze(trainMatYAll)).float()
X_test = torch.from_numpy(testMatX).float()

X = Xt.clone().detach().requires_grad_(True)
Y = Yt.clone().detach().requires_grad_(True)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = myGPs.thinPlateModel(X, Y, likelihood)

hypers = {
    'likelihood.noise_covar.noise': torch.tensor(0.1),
    'covar_module.max_dist': torch.tensor(0.8),
}
model.initialize(**hypers)

model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.4)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

print('\n\n**Printing all model constraints...**\n')
for constraint_name, constraint in model.named_constraints():
    print(f'Constraint name: {constraint_name:55} constraint = {constraint}')

print('\n\n**Printing all model parameters...**\n')
for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param.item()}')
print('\n')

training_iter = 4
for i in range(training_iter):

    optimizer.zero_grad()
    output = model(X)
    loss = -mll(output, Y)
    loss.backward()
    optimizer.step()
    print("Iteration: ",i)
    print("Loss: ",loss.item())   
    print(f'Actual likelihoood noise covariance: {likelihood.noise_covar.noise.item()}')
    print(f'Actual maximum distance: {model.covar_module.max_dist.item()}')
    print('\n')
     
model.eval() 
likelihood.eval()

# Get into evaluation (predictive posterior) mode
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(X_test))
    mean = observed_pred.mean
    #lower, upper = observed_pred.confidence_region()

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    # Get upper and lower confidence bounds
    #lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(mean.numpy())
    #ax.plot(lower.numpy())
    #ax.plot(upper.numpy())
    plt.show()

mu = mean.numpy()
#indexes = np.absolute(mu)<0.0004
indexes = np.absolute(mu)<0.005
with torch.no_grad():
    fig3D = plt.figure(figsize=plt.figaspect(1))  
    ax = fig3D.gca(projection='3d')
    ax.scatter(trainMatXAll[0:trainN,0], trainMatXAll[0:trainN,1], trainMatXAll[0:trainN,2], color='g')
    #ax.scatter(testMatX[:,0], testMatX[:,1], testMatX[:,2])
    ax.scatter(testMatX[indexes,0], testMatX[indexes,1], testMatX[indexes,2]) #, c=mu[indexes] , cmap='cool')
    #ax.scatter(testMatX[:,0], testMatX[:,1], testMatX[:,2]) #, c=mu[indexes] , cmap='cool')
    plt.show()
