# Define a superquadric
ni = np.linspace(0, np.pi, N)
omega = np.linspace(0, 2*np.pi, N)

eps1= 1.2
eps2= 1.2
alpha1= 0.5
alpha2= 0.5
alpha3= 0.5

xs = alpha1*np.float_power(np.cos(ni),eps1)*np.float_power(np.cos(omega),eps2)
ys = alpha2*np.float_power(np.cos(ni),eps1)*np.float_power(np.sin(omega),eps2)
zs = alpha3*np.float_power(np.sin(ni),eps2)
Xtest = np.column_stack([xs.ravel(),ys.ravel()])

# Define an ellipsoid 
u = np.linspace(0, 2 * np.pi, N)
v = np.linspace(0, np.pi, N)

a = 1
b = 2 
c = 1
x = a * np.outer(np.cos(u), np.sin(v))
y = b * np.outer(np.sin(u), np.sin(v))
z = c * np.outer(np.ones_like(u), np.cos(v))

x = x.flatten()
y = y.flatten()
z = z.flatten()


def superquadricImplicit(superParam,point,eps):
    
    feval = ((point[0]/superParam["a1"])**(2/superParam["lb5"]) + (point[1]/superParam["a2"])**(2/superParam["lb5"]))**(superParam["lb5"]/superParam["lb4"]) + (point[2]/superParam["a3"])**(2/superParam["lb4"])
    if feval < -eps:
        return -1
    elif feval > eps:
        return 1
    else:
        return 0


def superquadricExplicit(superParam,point,eps):
    
    ni = np.linspace(-np.pi/2, np.pi/2, N)
    omega = np.linspace(-np.pi, np.pi, N)
    x = superParam["a1"] * np.outer(np.cos(u), np.sin(v))
    y = superParam["a2"] * np.outer(np.sin(u), np.sin(v))
    z = superParam["a3"] * np.outer(np.ones_like(u), np.cos(v))

    feval = ((point[:,0]/superParam["a1"])**(2/superParam["lb5"]) + (point[:,1]/superParam["a2"])**(2/superParam["lb5"]))**(superParam["lb5"]/superParam["lb4"]) + (point[:,2]/superParam["a3"])**(2/superParam["lb4"])
    if feval < -eps:
        return -1
    elif feval > eps:
        return 1
    else:
        return 0

        dlim = 5
nTrainX, nTrainY, nTrainZ = (10, 10, 10)
nTestX, nTestY, nTestZ = (6, 6, 6)
l = 1
varS = 0.1
varN = 4

xTest, yTest, zTest= np.meshgrid(np.linspace(-dlim, dlim, nTestX),np.linspace(-dlim, dlim, nTestY),np.linspace(-dlim, dlim, nTestZ))
x = xTest.ravel()
y = yTest.ravel()
z = zTest.ravel()
testMatX = np.column_stack([x,y,z])
print("testMatX: ", testMatX.shape)

xTrain, yTrain, zTrain = np.meshgrid(np.linspace(-dlim, dlim, nTrainX),np.linspace(-dlim, dlim, nTrainY),np.linspace(-dlim, dlim, nTrainZ))
x = xTrain.ravel()
y = yTrain.ravel()
z = zTrain.ravel()
trainMatX = np.column_stack([x,y,z])
trainMatY = trainMatX

superParam = {"a1": 10.0, "a2": 10.0, "a3": 10.0, "lb4": 0.1, "lb5": 0.1}
index = 0
for point in trainMatX:
    trainMatY[index] = superquadricImplicit(superParam,trainMatX[index,:],0.01)
    print(superquadricImplicit(superParam,trainMatX[index,:],0.01))
    index = index +1
    

print("trainMatX:", trainMatX.shape)
#print("trainMatY:", trainMatY.shape)

fig3D = plt.figure(figsize=plt.figaspect(1))  # Square figure
ax = fig3D.gca(projection='3d')
ax.scatter(xTrain, yTrain, trainMatY.reshape(nTrainX,nTrainY), color='r')
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-dlim, dlim))

getattr(ax, 'set_{}lim'.format('z'))((-1, 1))
plt.show()



