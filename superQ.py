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
