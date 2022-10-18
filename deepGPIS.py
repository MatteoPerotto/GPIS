import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import urllib.request
import os
import numpy as np 
from math import floor

import time
import gpdata 
import myGPs


path = "../partial_pc/002_master_chef_can_0000_pc.pcd"
mesh_path = "../YCB_Video_Models/models/002_master_chef_can/textured_simple.obj"
complete_pc_path = "/home/user/YCB_Video_Models/models/002_master_chef_can/points.xyz"
T = np.array([[-0.70249935,  0.41384923, -0.57898487,  0.4 ],
                [ 0.71168439,  0.40850807, -0.57151246,  0.4 ],
                [ 0.,         -0.81354162, -0.58150669,  0.4 ],
                [ 0.,          0.,          0.,          1. ]])

trainP = 200
testP = 70000
outDim = 0.01
dataHandler = gpdata.GPdataHandler(complete_pc_path, trainP, testP)
trainMatXAll, trainMatYAll = dataHandler.genFromNormals(outDim, distanceLabel=True)
testMatX = dataHandler.genTestPoints()

#trainMatXAll, trainMatYAll = dataHandler.addTactilePoints(trainMatXAll, trainMatYAll, mesh_path, T, num = 4, rep = 20, outDim=outDim)

# Torch tensors  
Xt = torch.from_numpy(trainMatXAll).float()
Yt = torch.from_numpy(np.squeeze(trainMatYAll)).float()

X = Xt.clone().detach().requires_grad_(True)
y = Yt.clone().detach().requires_grad_(True)

r = torch.randperm(X.size()[0])
X = X[r]
y = y[r]

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()
trainX0 = train_x[(train_y==0).nonzero().squeeze(1)]

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

predictionX = torch.from_numpy(np.squeeze(testMatX)).float()

# Cuda 
#if torch.cuda.is_available():
#    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

data_dim = train_x.size(-1)
class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 500)) #1000
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(500, 250)) #500
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(250, 50)) #250
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, 5)) # 50 

feature_extractor = LargeFeatureExtractor()

class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = myGPs.ThinPlateRegularizer()
            self.feature_extractor = feature_extractor
            #self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

        def forward(self, x):

            # Feature extractor
            projected_x = self.feature_extractor(x)
            # Scale to bounds 
            #projected_x = self.scale_to_bounds(projected_x)  

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

hypers = {
    'likelihood.noise_covar.noise': torch.tensor(1e-4), #1e-4
    'covar_module.max_dist': torch.tensor(0.25), #0.5
}
model.initialize(**hypers)

#if torch.cuda.is_available():
#    model = model.cuda()
#    likelihood = likelihood.cuda()

model.train()
likelihood.train()

for name, param in model.named_parameters():
    print(name)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 
#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50])

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
iterations = 200

with gpytorch.settings.max_cholesky_size(0):
    for it in range(iterations):
        # Zero backprop gradients
        optimizer.zero_grad()
        # Get output from model
        output = model(train_x)
        # Calc loss and backprop derivatives
        loss = -mll(output, train_y)
        loss.backward(retain_graph=True)
        optimizer.step()
        #optimizer.step(closure)
        print("Iteration: ",it)
        print("Loss: ",loss.item())   
        print(f'Actual likelihoood noise covariance: {likelihood.noise_covar.noise.item()}')
        print(f'Actual maximum distance: {model.covar_module.max_dist.item()}')
        print('\n')
        scheduler.step()

    model.eval()
    likelihood.eval()

    ## Evaluation 
    with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
        f_preds = model(test_x)
        f_var = f_preds.variance
        y_preds = likelihood(model(test_x))  
        y_var = y_preds.variance 
        print('Test MAE: {}'.format(torch.mean(torch.abs(y_preds.mean - test_y))))   


        preds = likelihood(model(predictionX))
        var = preds.variance
        mu = preds.mean

        lower =  mu - var
        upper =  mu + var
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(mu.numpy())
        ax.plot(lower.numpy())
        ax.plot(upper.numpy())
        plt.show()

        indexes = np.absolute(mu)<1e-3
        fig3D = plt.figure(figsize=plt.figaspect(1))  
        ax = fig3D.gca(projection='3d')
        ax.scatter(trainX0[:,0], trainX0[:,1], trainX0[:,2], color='g')
        ax.scatter(predictionX[indexes,0], predictionX[indexes,1], predictionX[indexes,2], c=var[indexes])
        plt.show()
    