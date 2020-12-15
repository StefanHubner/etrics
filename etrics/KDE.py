from etrics.Utilities import EventHook, Timing

import scipy as sp
import numpy as np

import statsmodels.api as sm

def tensorprod(d, fcts):
    if d.shape[1] == len(fcts):
        return np.product([ fcts[i](d[:,i]) for i in range(len(fcts)) ], axis=0)
    else:
        raise Exception()

class KDE:
    def __init__(self, endog, **kwargs):
        self.endog = endog
        self.n = endog.shape[0]
        self.d = endog.shape[1]
        self.bwc = 1.5

    def bw(self, derivative):
        return {True:0.97, False:1.06}[derivative] * np.diag(np.std(self.endog, axis=0)) * self.n ** (-1/(4+self.d))

    def pdf_(self, y0, kernels, der = False):
        return 1/(np.linalg.det(self.bw(der)) * self.n) * np.sum(tensorprod(np.dot((self.endog - y0), np.linalg.inv(self.bw(der))), kernels))

    def pdf(self, y0):
        return self.pdf_(y0, [sp.stats.norm.pdf] * self.endog.shape[1])

    def dpdf(self, y0, idx):
        ks = [sp.stats.norm.pdf] * self.endog.shape[1]
        ks[idx] = lambda y: y * sp.stats.norm.pdf(y)
        return self.pdf_(y0, ks, der = True) / self.bw(True)[idx,idx]

class KDEBuilder:
    def __init__(self, **kwargs):
        self.name_idcs = {}
        self.endog = np.array([])
        for k,v in kwargs.items():
            self.endog = np.hstack([self.endog, v]) if self.endog.size else v
            self.name_idcs[k] = list(range(self.endog.shape[1])[-v.shape[1]:])

    def idcs(self, l): 
        return sum([self.name_idcs[k] for k in l], []) # sum over [] flattens (like foldr)

    def build_density(self, base):
        def f(y0):
            e = KDE(self.endog[:,self.idcs(base)])
            return e.pdf(y0)
        return f

    def build_derivative(self, base, derivative):
        def df(y0):
            e = KDE(self.endog[:,self.idcs(base)])
            # the derivative index is relative to the given base which is sometimes not the overall base (e.g. base = cond in dcf1)
            idcs1 = np.where([bi in self.idcs(derivative) for bi in self.idcs(base)])[0]
            return np.array(list(map(lambda i: e.dpdf(y0, i), idcs1)))
        return df

    def build_cdensity(self, base, cond):
        def cf(y0x0, x0):
            num = self.build_density(base)
            den = self.build_density(cond)
            return num(y0x0)/den(x0)
        return cf

    def build_cderivative(self, base, cond, derivative):
        def dcf1(y0x0, x0):
            du = self.build_derivative(base, derivative)
            v = self.build_density(cond)
            u = self.build_density(base)
            dv = self.build_derivative(cond, derivative)
            return (du(y0x0) * v(x0) - u(y0x0) * dv(x0)) / (v(x0) ** 2)
        def dcf2(y0x0, x0):
            num = self.build_derivative(base, derivative)
            den = self.build_density(cond)
            return num(y0x0)/den(x0)

        return dcf1 if all([v in cond for v in derivative]) else dcf2

    def build(self, s): # will take sth like "XY|Z_Z"
        import re
        import functools
        args = dict(map(lambda a: (a[0], list(a[1:])), re.findall("[>|_]+[A-Za-z]*", ">{}".format(s))))
        if len(args) == 3:
            return self.build_cderivative(args['>'], args['|'], args['_'])
        elif len(args) == 2:
            if '|' in args.keys():
                return self.build_cdensity(args['>'], args['|'])
            else:
                return self.build_derivative(args['>'], args['_'])
        else: 
            return self.build_density(args['>'])

if __name__ == "main":
    import numpy as np
    import KDE
    from KDE import KDE, KDEBuilder
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
    import matplotlib.pyplot as plt
    n = 1000000
    dim = 6
    d = np.random.normal(0, 1, size=n*dim).reshape(n,dim)
    Y, X, Z = d[:,[0,1]], d[:,[2,3]], d[:,4:dim]

    b = KDEBuilder(Y = Y, X = X, Z = Z)
    dens = b.build("Y")
    ddens = b.build("Y_Y")
    b.build("YX|X")(np.ones(4), np.ones(2))
    b.build("YXZ|XZ")(np.ones(6), np.ones(4))
    dcdens = b.build("YXZ|XZ_Z")(np.ones(6), np.ones(4))

    y0s = np.linspace(np.min(d), np.max(d))
    y = lambda f: np.array(list(map(lambda y1: f(np.array([y1, 0])), y0s)))
    plt.plot(y0s, y(dens), y0s, y(lambda y: ddens(y)[0]))
