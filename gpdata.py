import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class GPdataHandler():

    def __init__(self, pcdPoints, pcdCenter, trainN, centered=True):

        self.pointN = pcdPoints.shape[0]
        self.trainN = trainN
        mask = np.zeros(self.pointN, dtype=bool)
        mask[np.random.choice(np.arange(0,self.pointN), size=trainN, replace=False)] = True
        self.mask = mask
        self.pcdCenter = pcdCenter
        self.centered = centered

        if pcdPoints.shape[1] == 3 and pcdPoints.shape[0] > 0:
            self.trainMatX = np.column_stack([pcdPoints[mask,0],pcdPoints[mask,1],pcdPoints[mask,2]])
            self.trainMatY = np.zeros((trainN,1))
        else:
            print("[INITIALIZATION ERROR] Point clouds must be formatted as numpy array of shape (n,3)")
    
    def genFromNormals(self, pcdNormals, outDim):

        trainMatXOutside = np.zeros((self.trainN,3))
        trainMatYOutside = np.ones((self.trainN,1))

        trainMatXInside = np.zeros((self.trainN,3))
        trainMatYInside = (-1)*np.ones((self.trainN,1))

        # Mask normals 
        pcdNormals = pcdNormals[self.mask]

        # Add point inside and outside 
        for index,normal in enumerate(pcdNormals):
            xOut = self.trainMatX[index,0] + outDim*normal[0] 
            yOut = self.trainMatX[index,1] + outDim*normal[1] 
            zOut = self.trainMatX[index,2] + outDim*normal[2] 
            trainMatXOutside[index,:] = np.array([xOut,yOut,zOut])
            xIn = self.trainMatX[index,0] + -outDim*normal[0]
            yIn = self.trainMatX[index,1] + -outDim*normal[1]
            zIn = self.trainMatX[index,2] + -outDim*normal[2]
            trainMatXInside[index,:] = np.array([xIn,yIn,zIn])
        
        trainMatXAll = np.row_stack((self.trainMatX,trainMatXOutside,trainMatXInside))
        trainMatYAll = np.row_stack((self.trainMatY,trainMatYOutside,trainMatYInside))

        if self.centered == True:
            trainMatXAll = trainMatXAll - self.pcdCenter 

        #fig3D = plt.figure(figsize=plt.figaspect(1))  
        #ax = fig3D.gca(projection='3d')
        #ax.scatter(self.trainMatX[:,0], self.trainMatX[:,1], self.trainMatX[:,2], color='g')
        #ax.scatter(trainMatXOutside[:,0], trainMatXOutside[:,1], trainMatXOutside[:,2], color='r')
        #ax.scatter(trainMatXInside[:,0], trainMatXInside[:,1], trainMatXInside[:,2], color='b')
        #plt.show()

        return trainMatXAll, trainMatYAll
 
    def genFromBB(self, boxCenter, boxPoints, boxDim, spherePointN=0):
        
        trainMatXAll = np.concatenate((self.trainMatX, boxPoints), axis=0)
        trainMatYAll = np.concatenate((self.trainMatY, np.ones((boxPoints.shape[0],1))))

        center = np.reshape(boxCenter, (1, 3)) 
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
                testPointSphere[i,:]= self.pcdCenter + np.array([x, y, z])*0.01
            trainMatXAll = np.concatenate((trainMatXAll,testPointSphere), axis=0)
            trainMatYAll = np.concatenate((trainMatYAll, (-1)*np.ones((testPointSphere.shape[0],1))))
      

        if self.centered == True:
            trainMatXAll = trainMatXAll - self.pcdCenter 

        fig3D = plt.figure(figsize=plt.figaspect(1))  
        ax = fig3D.gca(projection='3d')
        ax.scatter(self.trainMatX[:,0], self.trainMatX[:,1], self.trainMatX[:,2], color='g')
        ax.scatter(boxPoints[:,0], boxPoints[:,1], boxPoints[:,2], color='r')
        ax.scatter(boxCenter[0], boxCenter[1], boxCenter[2], color='b')
        ax.scatter(testPointSphere[:,0], testPointSphere[:,1], testPointSphere[:,2], color='b')
        plt.show()

        return trainMatXAll, trainMatYAll

    def genTestPoints(self, testPointN, boxCenter, boxDim):

        boxEdgePoints = int(pow(testPointN,1/3))

        ## Create test points 
        xV = np.linspace(boxCenter[0]-boxDim[0],boxCenter[0]+boxDim[0],boxEdgePoints)
        yV = np.linspace(boxCenter[1]-boxDim[1],boxCenter[1]+boxDim[1],boxEdgePoints)
        zV = np.linspace(boxCenter[2]-boxDim[2],boxCenter[2]+boxDim[2],boxEdgePoints)

        x, y, z = np.meshgrid(xV, yV, zV)
        testMatX = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

        if self.centered == True:
            testMatX = testMatX - self.pcdCenter

        #fig3D = plt.figure(figsize=plt.figaspect(1))  
        #ax = fig3D.gca(projection='3d')
        #ax.scatter(testMatX[:,0], testMatX[:,1], testMatX[:,2], color='g')
        #plt.show()

        return testMatX











    

