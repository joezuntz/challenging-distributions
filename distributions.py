import emcee
import numpy as np
import pyDOE  # for latin hypercube
import contextlib
from scipy.special import logsumexp

# https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)



class Distribution:

    def emcee_test(self, samples=10000):
        # Set up walkers and sampler
        walkers = max(16, self.ndim * 4)

        # Starting point around zero
        p0 = emcee.utils.sample_ball(np.zeros(self.ndim), np.zeros(self.ndim)+0.01, walkers)

        # Run the actual sampler
        sampler = emcee.EnsembleSampler(walkers, self.ndim, self)
        sampler.run_mcmc(p0, samples)

        # Save output results to file with standard name, both the chain and the log_post columns
        chain = out = np.hstack([sampler.flatchain, sampler.flatlnprobability.reshape(sampler.flatlnprobability.size,1)])
        np.savetxt(f"{self.__class__.__name__}_{self.ndim}.txt", chain)


class GaussianShell(Distribution):
    def __init__(self, ndim):
        self.ndim = ndim

        # radii and widths of the two shells, both the same
        self.r = 2.0
        self.w = 0.1

        # centres of the two shells
        self.c = [
            np.array([-3.5] + [0.0 for i in range(ndim-1)]),
            np.array([+3.5] + [0.0 for i in range(ndim-1)])
        ]


    def __call__(self, x):
        # from eqns 32 and 33 of https://arxiv.org/pdf/0809.3437.pdf
        d0 = ((x-self.c[0])**2).sum()**0.5
        d1 = ((x-self.c[1])**2).sum()**0.5
        p0 = -(d0 - self.r)**2 / 2 / self.w**2
        p1 = -(d1 - self.r)**2 / 2 / self.w**2
        # return logsumexp([p0,p1])
        return np.log(np.exp(p0) + np.exp(p1))


class Torus(Distribution):
    "Torus in 3D and Gaussian in higher dimensions"
    def __init__(self, ndim):
        self.R = 2.0 # radius to centre of Torus
        self.a = 0.2 # radius of Torus
        self.ndim = ndim

    def __call__(self, X):
        x, y, z = X[:3]
        r = np.sqrt(x**2 + y**2)
        t2 = (self.R - r)**2 + z**2
        p = -t2 / self.a**2
    
        if self.ndim > 3:
            p -= (X[3:]**2).sum() / self.a**2

        return p



class Rosenbrock(Distribution):
    "Torus in 3D and Gaussian in higher dimensions"
    def __init__(self):
        self.a = 1.0
        self.b = 100.0
        self.ndim = 2

    def __call__(self, X):
        x, y = X
        return - (self.a - x)**2  - self.b*(y - x**2)**2


class MultiModal(Distribution):
    "Gaussians, same number of modes as dimensions, scattered centres"
    def __init__(self, ndim, seed=12345):
        self.ndim = ndim
        self.w = 0.5
        d = 0.9*20 / ndim
        x0 = -9.0
        with temp_seed(seed):
            self.centres = (pyDOE.lhs(ndim)-0.5)*18  # centres from -9 .. 9 

    def __call__(self, x):
        p = 0
        p = [-0.5*np.linalg.norm(x-self.centres[i])**2 / self.w**2 for i in range(self.ndim)]
        p = logsumexp(p)
        return p



class Exponential(Distribution):
    def __init__(self, ndim):
        self.ndim = ndim

    def __call__(self, x):
        if np.any(x<0):
            return -np.inf
        return -np.linalg.norm(x) / 0.5


def main():
    Torus(3).emcee_test()
    Torus(4).emcee_test()
    GaussianShell(2).emcee_test()
    GaussianShell(5).emcee_test() 
    GaussianShell(10).emcee_test()
    MultiModal(10).emcee_test()
    Exponential(4).emcee_test()
    Rosenbrock().emcee_test()


if __name__ == '__main__':
    main()