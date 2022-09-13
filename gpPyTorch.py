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

import time

# Import point cloud
pcd = o3d.io.read_point_cloud("../3d-tools/python/mesh_pc_render/partial_pc/002_master_chef_can_0000_pc.pcd")
pcdPoints = np.asarray(pcd.points)
#maxPointZ = np.argmax(pcdPoints[:,2])
#points = [[0,0,0],pcdPoints[maxPointZ,:]]
#lines = [[0,1]]
#line_set = o3d.geometry.LineSet()
#line_set.points = o3d.utility.Vector3dVector(points)
#line_set.lines = o3d.utility.Vector2iVector(lines)
#o3d.visualization.draw_geometries([pcd,line_set])
#o3d.visualization.draw_geometries([pcd])

# Import original mesh 
mesh = o3d.io.read_triangle_mesh("../YCB_Video_Models/models/002_master_chef_can/textured_simple.obj")
T = np.array([[-0.70249935,  0.41384923, -0.57898487,  0.4 ],
                [ 0.71168439,  0.40850807, -0.57151246,  0.4 ],
                [ 0.,         -0.81354162, -0.58150669,  0.4 ],
                [ 0.,          0.,          0.,          1. ]])

mesh.transform(np.linalg.inv(T))
originalPcd = mesh.sample_points_uniformly(number_of_points=500)

# Select contact points 
maxZ = np.amax(pcdPoints[:,2])
origPcd = np.asarray(originalPcd.points)
mask = origPcd[:,2] > maxZ
origPcd = origPcd[mask]
dist = np.zeros(origPcd.shape[0])
for index,point in enumerate(origPcd):
    dist[index] = np.abs(np.dot([1,0,0],point))

sortedindexes = np.argsort(dist)
sortedPoints = origPcd[sortedindexes,:]
tactilePoints = sortedPoints[0:4,:]

touchP = o3d.geometry.PointCloud()
touchP.points = o3d.utility.Vector3dVector(tactilePoints)
touchP.paint_uniform_color([1,0, 0])
o3d.visualization.draw_geometries([originalPcd,pcd,touchP])

# Sort point cloud 
# pcdPoints=pcdPoints[pcdPoints[:, 2].argsort()]

# Center it  
pcdCenter = pcd.get_center()

# Estimate normals Reconstruction of the surface results to be as:
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd)
pcdNormals = np.asarray(pcd.normals)
#o3d.visualization.draw_geometries([pcd],point_show_normal=True)

# Estimate grid bounding box 
gridBoundingBox = o3d.geometry.PointCloud.get_oriented_bounding_box(pcd)
gridBoundingBox.scale(1.5,gridBoundingBox.center)
#gridBoundingBox.color = np.asarray([0, 0, 0])
#o3d.visualization.draw_geometries([pcd,gridBoundingBox])
boxCenter = gridBoundingBox.center
boxDim = gridBoundingBox.extent
boxPoints = np.asarray(gridBoundingBox.get_box_points())
            
trainN = 300
dataHandler = gpdata.GPdataHandler(pcdPoints, pcdCenter, trainN)
trainMatXAll, trainMatYAll = dataHandler.genFromNormals(pcdNormals, 0.01, distanceLabel=False)
#trainMatXAll, trainMatYAll = dataHandler.genFromBB(boxCenter, boxPoints, boxDim, boxEdgePointN=5, spherePointN=0,sphereRadius=0.01)

# Generate test points (no bounding box scaling)
boundingBox  = o3d.geometry.PointCloud.get_oriented_bounding_box(pcd)
#boundingBox.color = np.asarray([0, 0, 0])
#o3d.visualization.draw_geometries([pcd,boundingBox])
boxCenter = boundingBox.center
boxDim = boundingBox.extent
testPointN = 200000
testMatX = dataHandler.genTestPoints(testPointN,boxCenter,boxDim)

Xt = torch.from_numpy(trainMatXAll).float()
Yt = torch.from_numpy(np.squeeze(trainMatYAll)).float()
X_test = torch.from_numpy(testMatX).float()

X = Xt.clone().detach().requires_grad_(True)
Y = Yt.clone().detach().requires_grad_(True)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = myGPs.thinPlateModel(X, Y, likelihood)

hypers = {
    'likelihood.noise_covar.noise': torch.tensor(0.01),
    'covar_module.max_dist': torch.tensor(4.0),
}
model.initialize(**hypers)

model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

print('\n\n**Printing all model constraints...**\n')
for constraint_name, constraint in model.named_constraints():
    print(f'Constraint name: {constraint_name:55} constraint = {constraint}')

print('\n\n**Printing all model parameters...**\n')
for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param.item()}')
print('\n')

training_iter = 30
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
    start = time.time()
    observed_pred = likelihood(model(X_test))
    mean = observed_pred.mean
    end = time.time()
    print(end - start)
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
indexes = np.absolute(mu)<0.001
with torch.no_grad():
    fig3D = plt.figure(figsize=plt.figaspect(1))  
    ax = fig3D.gca(projection='3d')
    ax.scatter(trainMatXAll[0:trainN,0], trainMatXAll[0:trainN,1], trainMatXAll[0:trainN,2], color='g')
    #ax.scatter(testMatX[:,0], testMatX[:,1], testMatX[:,2])
    ax.scatter(testMatX[indexes,0], testMatX[indexes,1], testMatX[indexes,2], color='r') #, c=mu[indexes] , cmap='cool')
    #ax.scatter(testMatX[:,0], testMatX[:,1], testMatX[:,2]) #, c=mu[indexes] , cmap='cool')
    plt.show()


# Training after contact points 
tactilePoints = tactilePoints-pcdCenter
tactilePoints = np.repeat(tactilePoints, 50, axis=0)
trainMatXAll = np.concatenate((trainMatXAll, tactilePoints), axis=0)
trainMatYAll = np.concatenate((trainMatYAll, np.zeros((tactilePoints.shape[0],1))))

Xt = torch.from_numpy(trainMatXAll).float()
Yt = torch.from_numpy(np.squeeze(trainMatYAll)).float()

X = Xt.clone().detach().requires_grad_(True)
Y = Yt.clone().detach().requires_grad_(True)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = myGPs.thinPlateModel(X, Y, likelihood)

hypers = {
    'likelihood.noise_covar.noise': torch.tensor(0.01),
    'covar_module.max_dist': torch.tensor(4.0),
}
model.initialize(**hypers)

model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

print('\n\n**Printing all model constraints...**\n')
for constraint_name, constraint in model.named_constraints():
    print(f'Constraint name: {constraint_name:55} constraint = {constraint}')

print('\n\n**Printing all model parameters...**\n')
for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param.item()}')
print('\n')

training_iter = 30
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
    start = time.time()
    observed_pred = likelihood(model(X_test))
    mean = observed_pred.mean
    end = time.time()
    print(end - start)
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
indexes = np.absolute(mu)<0.001
with torch.no_grad():
    fig3D = plt.figure(figsize=plt.figaspect(1))  
    ax = fig3D.gca(projection='3d')
    ax.scatter(trainMatXAll[0:trainN,0], trainMatXAll[0:trainN,1], trainMatXAll[0:trainN,2], color='g')
    ax.scatter(testMatX[indexes,0], testMatX[indexes,1], testMatX[indexes,2], color='r') #, c=mu[indexes] , cmap='cool')
    ax.scatter(tactilePoints[:,0], tactilePoints[:,1], tactilePoints[:,2], color='b')
    #ax.scatter(testMatX[:,0], testMatX[:,1], testMatX[:,2]) #, c=mu[indexes] , cmap='cool')
    plt.show()

radii = [0.005, 0.01, 0.02, 0.04]
xyz = np.zeros((np.size(testMatX[indexes,0]), 3))
xyz[:,0] = testMatX[indexes,0]
xyz[:,1] = testMatX[indexes,1]
xyz[:,2] = testMatX[indexes,2]

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(xyz)
pcd2.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd2)
pcd2.translate(pcdCenter)

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd2)

densities = np.asarray(densities)
density_colors = plt.get_cmap('plasma')(
    (densities - densities.min()) / (densities.max() - densities.min()))
density_colors = density_colors[:, :3]
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = mesh.vertices
density_mesh.triangles = mesh.triangles
density_mesh.triangle_normals = mesh.triangle_normals
density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
o3d.visualization.draw_geometries([originalPcd,pcd,density_mesh])

#rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#    pcd, o3d.utility.DoubleVector(radii))
#o3d.visualization.draw_geometries([pcd,rec_mesh])
