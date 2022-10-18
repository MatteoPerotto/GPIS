import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import open3d as o3d

class GPdataHandler():

    def __init__(self, path, trainN, testN, centered=True):

        pcd = o3d.io.read_point_cloud(path)
        pcdPoints = np.asarray(pcd.points)
        self.maxZ = np.amax(pcdPoints[:,2])

        self.centered = centered
        self.pcdCenter = pcd.get_center()
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd)
        self.pcdNormals = np.asarray(pcd.normals)
        pointN = pcdPoints.shape[0]

        self.trainN = trainN
        mask = np.zeros(pointN, dtype=bool)
        mask[np.random.choice(np.arange(0,pointN), size=trainN, replace=False)] = True
        self.mask = mask

        self.testN = testN
        gridBoundingBox = o3d.geometry.PointCloud.get_oriented_bounding_box(pcd)
        gridBoundingBox.scale(1.5,gridBoundingBox.center)
        self.boxCenter = gridBoundingBox.center
        self.boxDim = gridBoundingBox.extent
        self.boxPoints = np.asarray(gridBoundingBox.get_box_points())
        
        if pcdPoints.shape[1] == 3 and pcdPoints.shape[0] > 0:
            self.trainMatX = np.column_stack([pcdPoints[mask,0],pcdPoints[mask,1],pcdPoints[mask,2]])
            self.trainMatY = np.zeros((trainN,1))
        else:
            print("[INITIALIZATION ERROR] Point clouds must be formatted as numpy array of shape (n,3)")
    
    def genFromNormals(self, outDim, distanceLabel=False):

        trainMatXInside = np.zeros((self.trainN,3))
        trainMatXOutside = np.zeros((self.trainN,3))
        
        if distanceLabel==False:
            trainMatYInside = (-1)*np.ones((self.trainN,1))
            trainMatYOutside = np.ones((self.trainN,1))
        else:
            trainMatYInside = (-outDim)*np.ones((self.trainN,1))
            trainMatYOutside = outDim*np.ones((self.trainN,1))

        # Mask normals 
        pcdNormals = self.pcdNormals[self.mask]

        # Add point inside and outside 
        for index,normal in enumerate(pcdNormals):
            xOut = self.trainMatX[index,0] + outDim*normal[0] 
            yOut = self.trainMatX[index,1] + outDim*normal[1] 
            zOut = self.trainMatX[index,2] + outDim*normal[2] 
            trainMatXOutside[index,:] = np.array([xOut,yOut,zOut])
            xIn = self.trainMatX[index,0] - outDim*normal[0]
            yIn = self.trainMatX[index,1] - outDim*normal[1]
            zIn = self.trainMatX[index,2] - outDim*normal[2]
            trainMatXInside[index,:] = np.array([xIn,yIn,zIn])

        trainMatXAll = np.row_stack((self.trainMatX,trainMatXOutside,trainMatXInside))
        trainMatYAll = np.row_stack((self.trainMatY,trainMatYOutside,trainMatYInside))

        if self.centered == True:
            trainMatXAll = trainMatXAll - self.pcdCenter 

        fig3D = plt.figure(figsize=plt.figaspect(1))  
        ax = fig3D.gca(projection='3d')
        ax.scatter(self.trainMatX[:,0], self.trainMatX[:,1], self.trainMatX[:,2], color='g')
        ax.scatter(trainMatXOutside[:,0], trainMatXOutside[:,1], trainMatXOutside[:,2], color='r')
        ax.scatter(trainMatXInside[:,0], trainMatXInside[:,1], trainMatXInside[:,2], color='b')
        plt.show()
        
        return trainMatXAll, trainMatYAll
 
    def genFromBB(self, boxEdgePointN=0, spherePointN=0, sphereRadius=0.01):
        
        trainMatXAll = np.concatenate((self.trainMatX, self.boxPoints), axis=0)
        trainMatYAll = np.concatenate((self.trainMatY, np.ones((self.boxPoints.shape[0],1))))

        center = np.reshape(self.boxCenter, (1, 3)) 
        trainMatXAll = np.concatenate((trainMatXAll, center), axis=0)
        trainMatYAll = np.concatenate((trainMatYAll, (-1)*np.ones((center.shape[0],1))))
        
        if spherePointN !=0:    
            testPointSphere = np.zeros((spherePointN,3))
            phi = math.pi * (3. - math.sqrt(5.))  
            for i in range(spherePointN):
                y = 1 - (i / float(spherePointN - 1)) * 2  
                radius = math.sqrt(1 - y * y) 
                theta = phi * i  
                x = math.cos(theta) * radius
                z = math.sin(theta) * radius
                testPointSphere[i,:]= self.pcdCenter + np.array([x, y, z])*sphereRadius
            trainMatXAll = np.concatenate((trainMatXAll,testPointSphere), axis=0)
            trainMatYAll = np.concatenate((trainMatYAll, (-1)*np.ones((testPointSphere.shape[0],1))))
      
        if boxEdgePointN !=0:    
            edgePoints = np.zeros((28*boxEdgePointN,3))
            pBoxInd = 0
            for i1 in np.arange(0,8):
                for i2 in np.delete(np.arange(0,8),np.arange(0,i1+1)):
                    direction = np.subtract(self.boxPoints[i2],self.boxPoints[i1])
                    lenght = np.linalg.norm(direction)
                    for pInd in np.arange(1,boxEdgePointN+1):
                        if lenght<np.linalg.norm(self.boxDim):
                            edgePoints[pBoxInd,:] = self.boxPoints[i1]+direction*pInd/(boxEdgePointN+1)
                            pBoxInd += 1
            edgePoints = edgePoints[np.sum(edgePoints,axis=1)!=0]
            trainMatXAll = np.concatenate((trainMatXAll,edgePoints), axis=0)
            trainMatYAll = np.concatenate((trainMatYAll, np.ones((edgePoints.shape[0],1))))
 
        fig3D = plt.figure(figsize=plt.figaspect(1))  
        ax = fig3D.gca(projection='3d')
        ax.scatter(self.trainMatX[:,0], self.trainMatX[:,1], self.trainMatX[:,2], color='g')
        ax.scatter(self.boxPoints[:,0], self.boxPoints[:,1], self.boxPoints[:,2], color='r')
        ax.scatter(self.boxCenter[0], self.boxCenter[1], self.boxCenter[2], color='m')
        if spherePointN !=0 : ax.scatter(testPointSphere[:,0], testPointSphere[:,1], testPointSphere[:,2], color='b') 
        if boxEdgePointN !=0 : ax.scatter(edgePoints[:,0], edgePoints[:,1], edgePoints[:,2], color='c') 
        plt.show()

        return trainMatXAll, trainMatYAll    

    def addTactilePoints(self, trainMatXAll, trainMatYAll, complete_mesh_path, T, num, rep, outDim, distanceLabel=False):

        mesh = o3d.io.read_triangle_mesh(complete_mesh_path)
        mesh.transform(np.linalg.inv(T))
        originalPcd = mesh.sample_points_uniformly(number_of_points=1000)

        originalPcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        o3d.geometry.PointCloud.orient_normals_towards_camera_location(originalPcd)
        pcdNormals = np.asarray(originalPcd.normals)
       
        origPcd = np.asarray(originalPcd.points)
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
        
        if distanceLabel==False:
            tactileYInside = (-1)*np.ones((num,1))
            tactileYOutside = np.ones((num,1))
        else:
            tactileYInside = (-outDim)*np.ones((num,1))
            tactileYOutside = outDim*np.ones((num,1))
        
        for index,normal in enumerate(tactileNormals):
            xOut = tactilePoints[index,0] + outDim*normal[0] 
            yOut = tactilePoints[index,1] + outDim*normal[1] 
            zOut = tactilePoints[index,2] + outDim*normal[2] 
            tactileXOutside[index,:] = np.array([xOut,yOut,zOut])
            xIn = tactilePoints[index,0] - outDim*normal[0]
            yIn = tactilePoints[index,1] - outDim*normal[1]
            zIn = tactilePoints[index,2] - outDim*normal[2]
            tactileXInside[index,:] = np.array([xIn,yIn,zIn])

        #tactilePoints = np.repeat(tactilePoints, rep , axis=0)
        #tactileXInside = np.repeat(tactileXInside, rep , axis=0)
        #tactileXOutside = np.repeat(tactileXOutside, rep , axis=0)
        #tactileYInside = np.repeat(tactileYInside, rep , axis=0)
        #tactileYOutside = np.repeat(tactileYOutside, rep , axis=0)
        
        #print(tactilePoints.size)
        trainMatXTact = np.row_stack((trainMatXAll,tactilePoints,tactileXInside,tactileXOutside))
        trainMatYTact = np.row_stack((trainMatYAll,np.zeros((tactilePoints.size,1)),tactileYInside,tactileYOutside))

        fig3D = plt.figure(figsize=plt.figaspect(1))  
        ax = fig3D.gca(projection='3d')
        ax.scatter(self.trainMatX[:,0], self.trainMatX[:,1], self.trainMatX[:,2], color='g')
        ax.scatter(tactilePoints[:,0], tactilePoints[:,1], tactilePoints[:,2], color='b')
        ax.scatter(tactileXInside[:,0], tactileXInside[:,1], tactileXInside[:,2])
        ax.scatter(tactileXOutside[:,0], tactileXOutside[:,1], tactileXOutside[:,2])
        plt.show()

        return trainMatXTact, trainMatYTact
        
    def genTestPoints(self):

        boxEdgePoints = int(pow(self.testN,1/3))

        ## Create test points 
        xV = np.linspace(self.boxCenter[0]-self.boxDim[0],self.boxCenter[0]+self.boxDim[0],boxEdgePoints)
        yV = np.linspace(self.boxCenter[1]-self.boxDim[1],self.boxCenter[1]+self.boxDim[1],boxEdgePoints)
        zV = np.linspace(self.boxCenter[2]-self.boxDim[2],self.boxCenter[2]+self.boxDim[2],boxEdgePoints)

        x, y, z = np.meshgrid(xV, yV, zV)
        testMatX = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        if self.centered == True:
            testMatX = testMatX - self.pcdCenter

        return testMatX











    

