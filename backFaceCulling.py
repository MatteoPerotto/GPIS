import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def projectMesh(meshCoord,nx,ny,nz,angleLimit):

    cameraDir = np.array([nx,ny,nz]) 
    dot = np.dot(meshCoord[:,3:],cameraDir)

    return meshCoord[dot<-math.cos(angleLimit),:],meshCoord[dot>-math.cos(angleLimit),:]
    
with open("/home/user/GPIS/data/sugar.off") as file:
    points = file.readlines()
    pointN = len(points)
    meshCoord = np.zeros((pointN,6)) 
    for index,point in enumerate(points):
        if index > 1:
            values = point.split()
            meshCoord[index,0] = float(values[0])
            meshCoord[index,1] = float(values[1])
            meshCoord[index,2] = float(values[2])
            meshCoord[index,3] = float(values[3])
            meshCoord[index,4] = float(values[4])
            meshCoord[index,5] = float(values[5])

partialMesh,hiddenPoints = projectMesh(meshCoord,1.57,0,0,55)

fig3D = plt.figure(figsize=plt.figaspect(1))
ax = fig3D.gca(projection='3d')
#ax.scatter(meshCoord[:,0], meshCoord[:,1], meshCoord[:,2], color='b')
ax.scatter(partialMesh[:,0], partialMesh[:,1], partialMesh[:,2], color='r')
ax.scatter(hiddenPoints[:,0], hiddenPoints[:,1], hiddenPoints[:,2], color='g')
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-0.1, 0.1))
plt.show()