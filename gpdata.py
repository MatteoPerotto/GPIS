import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import open3d as o3d

class GPdataHandler():
    def __init__(self, pcd, trainN, outDim, distanceLabel=True):

        self.outDim = outDim
        self.distanceLabel = distanceLabel

        if not isinstance(pcd,o3d.geometry.PointCloud):
            pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd))

        self.pcdCenter = pcd.get_center()
        pcd.translate(-self.pcdCenter) 
        pcdPoints = np.asarray(pcd.points)
        self.maxZ = np.amax(pcdPoints[:,2])

        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        self.pcdNormals = np.asarray(pcd.normals)
        pointN = pcdPoints.shape[0]

        self.trainN = trainN
        mask = np.zeros(pointN, dtype=bool)
        mask[np.random.choice(np.arange(0,pointN), size=trainN, replace=False)] = True
        self.mask = mask

        gridBoundingBox = o3d.geometry.PointCloud.get_oriented_bounding_box(pcd)
        gridBoundingBox.scale(1.5,gridBoundingBox.center)
        self.boxCenter = gridBoundingBox.center
        self.boxDim = gridBoundingBox.extent
        self.boxDiag = np.sqrt(pow(self.boxDim[0],2)+pow(self.boxDim[1],2)+pow(self.boxDim[2],2))
        self.boxPoints = np.asarray(gridBoundingBox.get_box_points())
        
        if pcdPoints.shape[1] == 3 and pcdPoints.shape[0] > 0:
            self.trainMatX = np.column_stack([pcdPoints[mask,0],pcdPoints[mask,1],pcdPoints[mask,2]])
            self.trainMatY = np.zeros((trainN,1))
        else:
            print("[INITIALIZATION ERROR] Point clouds must be formatted as numpy array of shape (n,3)")
    
    def genFromNormals(self):

        trainMatXInside = np.zeros((self.trainN,3))
        trainMatXOutside = np.zeros((self.trainN,3))
        
        if self.distanceLabel == False:
            trainMatYInside = (-1)*np.ones((self.trainN,1))
            trainMatYOutside = np.ones((self.trainN,1))
        else:
            trainMatYInside = (-self.outDim)*np.ones((self.trainN,1))
            trainMatYOutside = self.outDim*np.ones((self.trainN,1))

        # Mask normals 
        pcdNormals = self.pcdNormals[self.mask]

        # Add point inside and outside 
        for index,normal in enumerate(pcdNormals):
            xOut = self.trainMatX[index,0] + self.outDim*normal[0] 
            yOut = self.trainMatX[index,1] + self.outDim*normal[1] 
            zOut = self.trainMatX[index,2] + self.outDim*normal[2] 
            trainMatXOutside[index,:] = np.array([xOut,yOut,zOut])
            xIn = self.trainMatX[index,0] - self.outDim*normal[0]
            yIn = self.trainMatX[index,1] - self.outDim*normal[1]
            zIn = self.trainMatX[index,2] - self.outDim*normal[2]
            trainMatXInside[index,:] = np.array([xIn,yIn,zIn])

        trainMatXAll = np.row_stack((self.trainMatX,trainMatXOutside,trainMatXInside))
        trainMatYAll = np.row_stack((self.trainMatY,trainMatYOutside,trainMatYInside))

        tX = o3d.geometry.PointCloud()
        tXin = o3d.geometry.PointCloud()
        tXout = o3d.geometry.PointCloud()
        tX.points = o3d.utility.Vector3dVector(self.trainMatX)
        tXin.points = o3d.utility.Vector3dVector(trainMatXInside)
        tXout.points = o3d.utility.Vector3dVector(trainMatXOutside)
        tX.paint_uniform_color([1,0,0])
        tXin.paint_uniform_color([0,1,0])
        tXout.paint_uniform_color([0,0,1])
        o3d.visualization.draw_geometries([tX,tXin,tXout])

        return trainMatXAll, trainMatYAll
 
    def addTactilePoints(self, trainMatXAll, trainMatYAll, pcd, num):
        
        pcd.translate(-self.pcdCenter)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcdNormals = np.asarray(pcd.normals)
       
        origPcd = np.asarray(pcd.points)
        mask = origPcd[:,2] > self.maxZ
        origPcd = origPcd[mask]
        pcdNormals = pcdNormals[mask]
        dist = np.zeros(origPcd.shape[0])
        for index,point in enumerate(origPcd):
            dist[index] = np.abs(np.dot([1,0,0],point))

        sortedindexes = np.argsort(dist)
        sortedPoints = origPcd[sortedindexes,:]
        tactilePoints = sortedPoints[0:num,:]

        tactileNormals = pcdNormals[sortedindexes,:]
        tactileNormals = tactileNormals[0:num,:]
    
        tactileXInside = np.zeros((num,3))
        tactileXOutside = np.zeros((num,3))
        
        if self.distanceLabel==False:
            tactileYInside = (-1)*np.ones((num,1))
            tactileYOutside = np.ones((num,1))
        else:
            tactileYInside = (-self.outDim)*np.ones((num,1))
            tactileYOutside = self.outDim*np.ones((num,1))
        
        for index,normal in enumerate(tactileNormals):
            xOut = tactilePoints[index,0] + self.outDim*normal[0] 
            yOut = tactilePoints[index,1] + self.outDim*normal[1] 
            zOut = tactilePoints[index,2] + self.outDim*normal[2] 
            tactileXOutside[index,:] = np.array([xOut,yOut,zOut])
            xIn = tactilePoints[index,0] - self.outDim*normal[0]
            yIn = tactilePoints[index,1] - self.outDim*normal[1]
            zIn = tactilePoints[index,2] - self.outDim*normal[2]
            tactileXInside[index,:] = np.array([xIn,yIn,zIn])
        
        tactilePoints = tactilePoints  
        tactileXInside = tactileXInside 
        tactileXOutside = tactileXOutside 

        trainMatXTact = np.row_stack((trainMatXAll,tactilePoints, tactileXInside, tactileXOutside))
        trainMatYTact = np.row_stack((trainMatYAll,np.zeros((tactilePoints.shape[0],1)),tactileYInside,tactileYOutside)) 

        tX = o3d.geometry.PointCloud()
        tXt = o3d.geometry.PointCloud()
        tXin = o3d.geometry.PointCloud()
        tXout = o3d.geometry.PointCloud()
        tX.points = o3d.utility.Vector3dVector(trainMatXAll)
        tXt.points = o3d.utility.Vector3dVector(tactilePoints)
        tXin.points = o3d.utility.Vector3dVector(tactileXInside)
        tXout.points = o3d.utility.Vector3dVector(tactileXOutside)
        tX.paint_uniform_color([1,0,0])
        tXt.paint_uniform_color([0,0,0])
        tXin.paint_uniform_color([0,1,0])
        tXout.paint_uniform_color([0,0,1])
        o3d.visualization.draw_geometries([tX,tXt,tXin,tXout])

        return trainMatXTact, trainMatYTact
        
    def genTestPoints(self, testN):

        boxEdgePoints = int(pow(testN,1/3))

        ## Create test points 
        xV = np.linspace(self.boxCenter[0]-self.boxDim[0],self.boxCenter[0]+self.boxDim[0],boxEdgePoints)
        yV = np.linspace(self.boxCenter[1]-self.boxDim[1],self.boxCenter[1]+self.boxDim[1],boxEdgePoints)
        zV = np.linspace(self.boxCenter[2]-self.boxDim[2],self.boxCenter[2]+self.boxDim[2],boxEdgePoints)

        x, y, z = np.meshgrid(xV, yV, zV)
        testMatX = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        return testMatX











    

