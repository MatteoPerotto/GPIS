import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import urllib.request
import os
import numpy as np 
from scipy.io import loadmat
from math import floor
import open3d as o3d

import gpdata 
import myGPs


pcd = o3d.io.read_point_cloud("../3d-tools/python/mesh_pc_render/partial_pc/002_master_chef_can_0000_pc.pcd")
pcdPoints = np.asarray(pcd.points)
 
pcdCenter = pcd.get_center()

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd)
pcdNormals = np.asarray(pcd.normals)
            
trainP = 500
dataHandler = gpdata.GPdataHandler(pcdPoints, pcdCenter, trainP)
trainMatXAll, trainMatYAll = dataHandler.genFromNormals(pcdNormals, 0.01, distanceLabel=False)

Xt = torch.from_numpy(trainMatXAll).float()
Yt = torch.from_numpy(np.squeeze(trainMatYAll)).float()

X = Xt.clone().detach().requires_grad_(True)
y = Yt.clone().detach().requires_grad_(True)

train_n = int(floor(0.8 * len(X)))
tr_x = X[:trainP, :].contiguous()
tst_x = X[trainP:, :].contiguous()

r = torch.randperm(X.size()[0])
X = X[r]
y = y[r]

train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

#if torch.cuda.is_available():
#    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

data_dim = train_x.size(-1)
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 500))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(500, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 2))

feature_extractor = LargeFeatureExtractor()

class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = myGPs.ThinPlateRegularizer()
            self.feature_extractor = feature_extractor

            # This module will scale the NN features so that they're nice values
            self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

        def forward(self, x):
            # We're first putting our data through a deep net (feature extractor)
            projected_x = self.feature_extractor(x)
            projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

hypers = {
    'likelihood.noise_covar.noise': torch.tensor(0.01),
    'covar_module.max_dist': torch.tensor(4.0),
}
model.initialize(**hypers)

#if torch.cuda.is_available():
#    model = model.cuda()
#    likelihood = likelihood.cuda()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters()},
    {'params': model.covar_module.parameters()},
    {'params': model.mean_module.parameters()},
    {'params': model.likelihood.parameters()},
], lr=0.8)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

iterator = range(60)
for i in iterator:
    # Zero backprop gradients
    optimizer.zero_grad()
    # Get output from model
    output = model(train_x)
    # Calc loss and backprop derivatives
    loss = -mll(output, train_y)
    loss.backward(retain_graph=True)
    optimizer.step()
    print("Iteration: ",i)
    print("Loss: ",loss.item())   
    print(f'Actual likelihoood noise covariance: {likelihood.noise_covar.noise.item()}')
    print(f'Actual maximum distance: {model.covar_module.max_dist.item()}')
    print('\n')

model.train()
likelihood.train()

model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
    preds = model(test_x)

print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    mean = preds.mean
    #lower, upper = preds.confidence_region()

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
indexes = np.absolute(mu)<0.05

with torch.no_grad():
    fig3D = plt.figure(figsize=plt.figaspect(1))  
    ax = fig3D.gca(projection='3d')
    ax.scatter(tr_x[:,0], tr_x[:,1], tr_x[:,2], color='g')
    ax.scatter(test_x[indexes,0], test_x[indexes,1], test_x[indexes,2], color='r')
    plt.show()


     
