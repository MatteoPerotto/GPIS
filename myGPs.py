import torch 
import gpytorch 

class thinPlateModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(thinPlateModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ThinPlateRegularizer()
        self.batch_shape = torch.Size([])
    # Forward computes the distribution fo the given input (multivariate normal)
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ThinPlateRegularizer(gpytorch.kernels.Kernel):
    # the sinc kernel is stationary
    is_stationary = True

    # We will register the parameter when initializing the kernel
    def __init__(self, dist_prior=None, dist_constraint=None, **kwargs):
        super().__init__(**kwargs)

        # register the raw parameter
        self.register_parameter(
            name='max_dist', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
        )

        # set the parameter constraint to be positive
        if dist_constraint is None:
            dist_constraint = gpytorch.constraints.GreaterThan(0)
           
        # register the constraint
        self.register_constraint("max_dist", dist_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if dist_prior is not None:
            self.register_prior(
                "dist_prior",
                dist_prior,
                lambda m: m.length,
                lambda m, v : m._set_length(v),
            )

    # now set up the 'actual' paramter
    @property
    def maxdist(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_dist_constraint.transform(self.max_dist)

    @maxdist.setter
    def maxdist(self, value):
        return self._set_maxdist(value)

    def _set_maxdist(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.max_dist)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(max_dist=self.raw_dist_constraint.inverse_transform(value))

    # Kernel function, define how to compute covariance matrix given two sets of points 
    def forward(self, x1, x2, diag=False ,**params):
        # calculate the distance between inputs
        diff = self.covar_dist(x1, x2,  diag=diag, **params)
        # prevent divide by 0 errors
        diff.where(diff == 0, torch.as_tensor(1e-20))
        tp = 2*torch.pow(diff,3)-3*self.max_dist*torch.pow(diff,2)+self.max_dist**3
        if diag:
            tp = tp[0]
        return tp

