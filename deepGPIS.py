import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import urllib.request
import os
import numpy as np 
from math import floor

import time

from torch.optim import Adam

import gpdata
import myGPs

import open3d as o3d
from utils.visualizer import Visualizer


def rotate_view(vis):
    ctr = vis.get_view_control()
    ctr.rotate(0.1, 0.0)
    return False

## Directly from point cloud 
#pcd = o3d.io.read_point_cloud("/home/user/YCB_Video_Models/models/002_master_chef_can/points.xyz")
#T = np.array([[-0.70249935,  0.41384923, -0.57898487,  0.4 ],
#                [ 0.71168439,  0.40850807, -0.57151246,  0.4 ],
#                [ 0.,         -0.81354162, -0.58150669,  0.4 ],
#                [ 0.,          0.,          0.,          1. ]])
#pcd = pcd.transform(np.linalg.inv(T))
#diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
#_, pt_map = pcd.hidden_point_removal([0,0,0], 1e3*diameter)
#visiblePcd = pcd.select_by_index(pt_map)
#hiddenPcd = pcd.select_by_index(pt_map, invert=True)
#visiblePcd.paint_uniform_color([1,0, 0])
#hiddenPcd.paint_uniform_color([0,0,1])
#o3d.visualization.draw_geometries([visiblePcd,hiddenPcd])

## More uniform sampling

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(3, 100))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(100, 50)) 
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(50, 10)) 

feature_extractor = LargeFeatureExtractor()

class GPRegressionModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = myGPs.ThinPlateRegularizer()
            self.feature_extractor = feature_extractor

            # self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)
            self.likelihood = likelihood
            #self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

        def forward(self, x):
            # Feature extractor
            projected_x = self.feature_extractor(x)
            # Scale to bounds 
            #projected_x = self.scale_to_bounds(projected_x)  

            mean_x = self.mean_module(projected_x)
            covar_x = self.covar_module(projected_x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        # def to_pc(self, itr, thr, num_points):
        #     device = 'cpu'
        #
        #     with torch.enable_grad():
        #         old = self.training
        #         self.eval()
        #
        #         refined_pred = torch.tensor(torch.randn(num_points * 2, 3).cpu().detach().numpy() * 1, device=device,
        #                                     requires_grad=True)
        #
        #         # loss_function = lambda x, gt: -self.mll(x, gt)
        #         optim = Adam([refined_pred], lr=0.1)
        #
        #         new_points = []
        #         # refined_pred.detach().clone()
        #         for step in range(itr):
        #             results = self(refined_pred)
        #
        #             idxs = self.likelihood(results).mean <= thr
        #             points = refined_pred.detach().clone()[idxs, :]
        #             new_points += [points]
        #
        #             gt = torch.ones_like(idxs, dtype=torch.float32)
        #             gt[:] = 1
        #             loss = loss_function(results, gt)
        #
        #             self.zero_grad()
        #             optim.zero_grad()
        #             loss.backward(inputs=[refined_pred])
        #             optim.step()
        #
        #     res = torch.cat(new_points)
        #
        #     perm = torch.randperm(res.size(0))
        #     res = res[perm[:num_points]]
        #
        #     return res[..., :3], res[..., -1]

def main():
    T = np.array([[-0.70249935, 0.41384923, -0.57898487, 0.4],
                  [0.71168439, 0.40850807, -0.57151246, 0.4],
                  [0., -0.81354162, -0.58150669, 0.4],
                  [0., 0., 0., 1.]])

    partial = True
    mesh = o3d.io.read_triangle_mesh("data/002_master_chef_can/google_16k/textured.obj", True)
    mesh = mesh.transform(np.linalg.inv(T))
    o3d.visualization.draw_geometries([mesh])

    pcd = mesh.sample_points_uniformly(number_of_points=2500)
    pcd = mesh.sample_points_poisson_disk(number_of_points=500, pcl=pcd)

    vis = o3d.visualization.Visualizer()

    if partial:
        diameter = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
        _, pt_map = pcd.hidden_point_removal([0, 0, 0], 1e3 * diameter)
        visiblePcd = pcd.select_by_index(pt_map)
        hiddenPcd = pcd.select_by_index(pt_map, invert=True)
        visiblePcd.paint_uniform_color([1, 0, 0])
        hiddenPcd.paint_uniform_color([0.5, 0.5, 0.5])
        vis.create_window()
        o3d.visualization.draw_geometries_with_animation_callback([visiblePcd], rotate_view)
        vis.destroy_window()

    trainN = 200
    testN = 150000
    outDim = 0.01  # 1 cm

    if partial:
        dataHandler = gpdata.GPdataHandler(visiblePcd, trainN, outDim, distanceLabel=True)
        trainMatXAll, trainMatYAll = dataHandler.genFromNormals()
        trainMatXAll, trainMatYAll = dataHandler.addTactilePoints(trainMatXAll, trainMatYAll, hiddenPcd, num=5)
    else:
        dataHandler = gpdata.GPdataHandler(pcd, trainN, outDim, distanceLabel=True)
        trainMatXAll, trainMatYAll = dataHandler.genFromNormals()

    testMatX = dataHandler.genTestPoints(testN)

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
    trainX0 = train_x[(train_y == 0).nonzero().squeeze(1)]

    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()

    predictionX = torch.from_numpy(np.squeeze(testMatX)).float()

    # Cuda
    # if torch.cuda.is_available():
    #    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

    data_dim = train_x.size(-1)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise_covar.register_constraint("raw_noise", gpytorch.constraints.Positive())

    model = GPRegressionModel(train_x, train_y, likelihood)

    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(1e-4), #1e-4
        'covar_module.max_dist': torch.tensor(dataHandler.boxDiag),
    }
    model.initialize(**hypers)

    #if torch.cuda.is_available():
    #    model = model.cuda()
    #    likelihood = likelihood.cuda()

    # model.train(True)
    # likelihood.train()

    for name, param in model.named_parameters():
        print(name)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.05) #0.05
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[25,50,100,200])

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    iterations = 300
    viewer = Visualizer()

    with gpytorch.settings.max_cholesky_size(0):
        for it in range(iterations):
            model.train()
            likelihood.train()

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
            y_preds = likelihood(model(test_x))
            # print('Test MAE: {}'.format(torch.mean(torch.abs(y_preds.mean - test_y))))

            preds = likelihood(model(predictionX))
            # predictionX = model.to_pc(20, 1e-3, 10_000)
            var = preds.variance
            mu = preds.mean

            indexes = ((np.absolute(mu)<1e-3).bool())# & (np.absolute(var)<1e-4).bool())
            fig3D = plt.figure(figsize=plt.figaspect(1))
            trainSet = o3d.geometry.PointCloud()
            trainSet.points = o3d.utility.Vector3dVector(trainX0)
            trainSet.paint_uniform_color([1,0,0])

            var = var[indexes].numpy()

            res = {'predictions': predictionX[indexes,:].cpu().numpy(), 'variance': var,
                   'train': trainMatXAll[trainMatYAll[:, 0] == 0]}
            viewer.update(res)


if __name__ == '__main__':
    main()