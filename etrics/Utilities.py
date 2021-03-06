import numpy as np
import logging
import pickle
import sys
from scipy.stats.distributions import triang
from scipy.stats import norm

class MaxHandlerException(Exception):
    def __init__(self, maxh):
        self.expr = "MaxHandlerException"
        self.message = "You may only add up to " + maxh + " event-handling delegates"

class EventHook(object):

    def __init__(self, maxhandlers=128):
        self.__handlers = []
        self.__maxhandlers = maxhandlers

    def __iadd__(self, handler):
        if len(self.__handlers) < self.__maxhandlers:
            self.__handlers.append(handler)
        else:
            raise MaxHandlerException(self.__maxhandlers)
        return self

    def __isub__(self, handler):
        self.__handlers.remove(handler)
        return self

    def Fire(self, *args, **keywargs):
        retvals = []
        for handler in self.__handlers:
            retvals.append(handler(*args, **keywargs))
        return retvals

    def ClearObjectHandlers(self, inObject):
        for theHandler in self.__handlers:
            if theHandler.im_self == inObject:
                self -= theHandler

    def SetMaxHandlers(self, maxh):
        self.__maxhandlers = maxh

    @property
    def NumHandlers(self):
        return len(self.__handlers)

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

class Timing:
    def __init__(self, name, trc = True, logger = None):
        self.name = name
        self.trc = trc
        self.logger = logger

    def __enter__(self):
        import time
        self.start = time.time()
        if self.trc:
            s = "{0}: ".format(self.name)
            if self.logger is None:
                print (s, end = "")
            else:
                self.logger.info(s)

            sys.stdout.flush()

    def __exit__(self, type_, value, traceback):
        import time, datetime
        self.stop = time.time()
        if self.trc:
            s = "{0}".format(str(datetime.timedelta(seconds=self.stop-self.start)))
            if self.logger is None:
                print (s)
            else:
                self.logger.info(s)

def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
    for j in range(1, arrays[0].size):
        out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

def vech(M):
    return np.array(np.matrix(M)[np.tril_indices(M.shape[0])])[0]

def varname(p):
    import inspect, re
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m: return m.group(1)

class multivnormal:
    def __init__(self, mean, variance):
        self._mean = mean
        self._variance = variance

    def rvs(self, n):
        return np.random.multivariate_normal(self._mean, self._variance, n)

    def ppf(self, tau):
        from scipy.stats.distributions import norm
        return [norm(self._mean[i], np.sqrt(self._variance[i][i])).ppf(tau) for i in range(len(self._mean))]

    @property
    def marginals(self):
        from scipy.stats.distributions import norm
        return [norm(self._mean[i], np.sqrt(self._variance[i][i])) for i in range(len(self._mean))]

class ConditionalDistributionWrapper:
    def __init__(self, basemarginals, inv, dinv):
        self.inv = inv
        self.dinv = dinv
        self.basemarginals = basemarginals

    def pdfjoint(self, x0, y):
        return np.prod(self.basepdfs(x0, y)) * np.abs(np.linalg.det(self.dinv(y, x0)))

    def pdf(self, y, x0):
        # TODO does that work like this with the marginals - if at all works ONLY if errors enter diagonally
        # print(self.basemarginals[0].pdf(.1), self.basemarginals[1].pdf(.05))
        # return self.basepdfs(x0, y) * np.abs(np.diagonal(self.dinv(y, x0))).T
        # this should work due to independence of epsilons, f_x1 = \int f(x1) f(x2) dx2 = J f_e1(xinv)
        # self.plotjoint(x0)
        # WARNING: this works only if epsilons enter diagonally, otherwise we need to obtain marginals from joint
        return self.basepdfs(x0, y) * np.abs(np.diagonal(self.dinv(y, x0)))

    def plotjoint(self, x0):
        #import matplotlib.pyplot as plt
        #from matplotlib import cm
        #from mpl_toolkits.mplot3d.axes3d import Axes3D
        #from etrics.Utilities import matrixbyelem
        #X, Y = np.meshgrid(np.linspace(20,36,100), np.linspace(10,26,100))
        #l = lambda y1, y2: self.pdfjoint(x0, np.array([y1, y2]))
        #Z = matrixbyelem(X, Y, l)
        #fig, ax = plt.subplots()
        #p = ax.pcolor(X, Y, Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), linewidth=0)
        #cb = fig.colorbar(p, ax=ax)
        #plt.show()
        from etrics.Utilities import quickjointdensityplot
        quickjointdensityplot(np.array([16,36]), np.array([10,26]), lambda y1, y2: self.pdfjoint(x0, np.array([y1, y2])))

    def basepdfs(self, x0, y):
        return np.array([f.pdf(e) for f,e in zip(self.basemarginals, self.inv(y, x0))])

class TriangularKernel:
    def __init__(self, scale, prepmoments=2):
        self.c = 0.5
        self.loc = -scale/2
        self.scale = scale
        fname = ".triangular_{0}".format(int(scale*100))
        import os.path
        if not os.path.isfile(fname): # only the first time, then save moments
            self.definesymbolic(scale)
            self.preparemoments(prepmoments)
            with open(fname, "wb") as f: pickle.dump(self.__dict__, f)
        else:
            with open(fname, "rb") as f: self.__dict__ = pickle.load(f)

    @property
    def support(self):
        return (self.loc, self.loc + self.scale)

    def definesymbolic(self, scale):
        from sympy import symbols, integrate, diff
        u, n, c = symbols("u n c")
        Kl = 2/scale + 4*u/scale**2
        Kr = 2/scale - 4*u/scale**2
        self._N  = integrate(Kl*u**n, (u, -scale/2, 0))
        self._N += integrate(Kr*u**n, (u, 0, +scale/2))
        self._T  = integrate(Kl**2*u**n, (u, -scale/2, 0))
        self._T += integrate(Kr**2*u**n, (u, 0, +scale/2))
        self._Q1  = integrate(diff(Kl,u)**2*u**n, (u, -scale/2, 0))
        self._Q1 += integrate(diff(Kr,u)**2*u**n, (u, 0, +scale/2))
        self._Q2 = c * integrate(Kl*u**(n-2), (u, -scale/2, 0))
        self._Q2 += c * integrate(Kr*u**(n-2), (u, 0, +scale/2))

    def preparemoments(self, nmom):
        from etrics.Utilities import matrixbyelem
        from sympy import Symbol
        I, J = np.meshgrid(range(nmom), range(nmom))
        self._Np = matrixbyelem(I, J, lambda i, j: self._N.subs(Symbol('n'), i+j).evalf())
        self._Npinv = np.linalg.inv(self._Np)
        self._Tp = matrixbyelem(I, J, lambda i, j: self._T.subs(Symbol('n'), i+j).evalf())
        self._Qp = matrixbyelem(I, J, lambda i, j: self._Q1.subs(Symbol('n'), i+j).evalf() - \
            (self._Q2.subs(Symbol('c'), 0.5*i*(i-1) + 0.5*j*(j-1)).subs(Symbol('n'), i+j).evalf() \
            if np.any(np.array([i,j])) > 1 else 0))
        self.momentsavailable = nmom

    @property
    def Np(self):
        # use parameter, call preparemoments if not enought moments available
        return self._Np

    @property
    def Npinv(self):
        return self._Npinv

    @property
    def Tp(self):
        return self._Tp

    @property
    def Qp(self):
        return self._Qp

    def pdf(self, y):
        return np.apply_along_axis(np.prod, 1, np.array(list(map(lambda yi: triang.pdf(yi, self.c, loc = self.loc, scale = self.scale), y))))

    def moment(self, n):
        return triang.moment(n, self.c, loc = self.loc, scale = self.scale)

    def moment2(self, n):
        return [1.0, 1.+ 4/3 + 1/2, (1/3 + 1/2 + 1/5)*self.scale][n]

    def momentprime2(self, n):
        return (np.array([16 , 4, 4/3]) * (self.scale ** np.array([-3,-2,-1])))[n]


# This is a product Kernel (assumes re-centering/rotation)
class StandardNormalKernel:
    def SetSigmas(self, sigmas):
        pass
    def pdf(self, y):
        def mannorm(yi):
            return (1/np.sqrt(2*np.pi))*np.exp(-0.5*yi**2)
        w = np.nan_to_num(np.array([np.product(mannorm(yi)) for yi in y]))
        return w

class TriangularKernel2:
    def SetSigmas(self, sigmas):
        self.c = 0.5
        self.sigmas = sigmas
        self.scales = 2 * np.sqrt(6) * sigmas
        self.locs = -self.scales/2
        #self.marginals = [triang(self.c, loc = l, scale = s) for l,s in zip(self.locs, self.scales)]

    def pdf(self, y):
        #marginals_old = [triang(self.c, loc = l, scale = s) for l,s in zip(self.locs, self.scales)]
        marginals = [lambda x: max(2.0/s - (4.0/(s**2))*abs(x), 0.0) for s in self.scales]
        ytmp = y if len(y.shape) == 2 else y.reshape((-1,1))
        #with Timing("eval pdfs", True): # triang.pdf very slow (1sec for N = 1500)
        #weights_old = np.apply_along_axis(lambda yi: np.product([m.pdf(yij) for m,yij in zip(marginals_old, yi)]), 1, ytmp)
        weights = np.apply_along_axis(lambda yi: np.product([m(yij) for m,yij in zip(marginals, yi)]), 1, ytmp)
        #print("Weights and old weights close? {}".format(np.all(np.isclose(weights, weights_old))))
        #print(weights[~np.isclose(weights, weights_old)])
        return weights

    def pdfnorm(self, y):
        # pickler problem with triang_gen
        marginals = [triang(self.c, loc = l, scale = s) for l,s in zip(self.locs/self.sigmas, self.scales/self.sigmas)] # have var's normalized to 1 (confirmed)
        ytmp = y if len(y.shape) == 2 else y.reshape((-1,1))
        return np.apply_along_axis(lambda yi: np.product([m.pdf(yij) for m,yij in zip(marginals, yi/self.sigmas)]), 1, ytmp)

    @property
    def B1(self):
        Npinv = np.linalg.inv(np.diag([1] + (self.sigmas**2).tolist() ))
        #Tp = np.diag([1] + ((self.scales/2)**2/6).tolist())
        # u,b = symbols("u b")
        # Kp = 1/b - 1/(b**2) * u
        # Kn = 1/b + 1/(b**2) * u
        # integrate(Kn**2 * u**2, (u,-b,0)) + integrate(Kp**2 * u**2, (u,0,b)) # b/15
        # integrate(Kn**2, (u,-b,0)) + integrate(Kp**2, (u,0,b)) # 2/(3*b)
        b = self.scales/2
        Tp = np.diag([np.product(2/(3*b))] + (b/15).tolist())

        return np.dot(np.dot(Npinv, Tp), Npinv)

def quickdensityplot( data):
    import pylab as plt
    from numpy import min, max
    import statsmodels.api as sm
    x = np.linspace(min(data), max(data))
    e = sm.nonparametric.KDEUnivariate(data)
    e.fit()
    plt.plot(x, e.evaluate(x))
    plt.show()

def quickjointdensityplot(y1, y2, dens = None, lim1 = None, lim2 = None, figax = None):
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from matplotlib import cm
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from etrics.Utilities import matrixbyelem
    X, Y = np.meshgrid(np.linspace(np.min(y1) if lim1 is None else lim1[0], isnp.max(y1) if lim1 is None else lim1[1], 100), np.linspace(np.min(y2) if lim2 is None else lim2[0], np.max(y2) if lim2 is None else lim2[1], 100))
    if dens is None:
        est = sm.nonparametric.KDEMultivariate(data=np.hstack([y1, y2]), var_type='cc', bw='normal_reference')
        dens = lambda y1, y2: est.pdf(np.array([y1, y2]))
    dens(X[0,0], Y[0,0])

    Z = matrixbyelem(X, Y, dens)
    fig, ax = plt.subplots() if figax is None else figax
    #p = ax.pcolor(X, Y, Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), linewidth=0, antialiased = True)
    p = ax.contourf(X, Y, Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), antialiased = True, levels=100)
    ax.plot(y1,y2, 'k.', markersize=2)

    ax.set_xlim(lim1)
    ax.set_ylim(lim2)
    #cb = fig.colorbar(p, ax=ax)
    if figax is None: plt.show()

    def basepdfs(self, x0, y):
        return np.array([f.pdf(e) for f,e in zip(self.basemarginals, self.inv(y, x0))])


def matrixbyelem(x, y, f):
    M =  np.ones(x.shape)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            M[i,j] = f(x[i,j], y[i,j])
    return M

class EstimatedDistWrapper:
    def __init__(self, est):
        self.e = est
    def pdf(self, x):
        return self.e.evaluate(x).item()

class UniformWeights:
    def __init__(self, k):
        self.k_vars = k
    def pdf(self, x):
        return (1.0)

class DegenerateDistribution:
        def __init__(self, const, K):
            self.const = const
            self.K = K

        def sample(self, N):
            #return np.array(np.matrix(np.repeat(self.const, N)).T)
            return np.repeat(self.const, N*self.K).reshape(N, self.K)

class EmpiricalDistribution:
    def __init__(self, data):
        self.data = data

    def cdf(self, est):
        raise NotImplementedError()

    def sample(self, N, truncateoutliers = True):
        idcs = list(map(int, np.floor(np.random.uniform(size=N)*self.data.shape[0])))
        if truncateoutliers:
            tails = 2
            filter_ = lambda x: np.maximum(np.minimum(np.percentile(self.data, 100-tails, axis=0), x), np.percentile(self.data, tails, axis=0))
        else:
            filter_ = lambda x: x
        return filter_(self.data[idcs,:])
