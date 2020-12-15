from etrics.NLRQ import NLRQ, Polynomial1, DPolynomial1, Polynomial2, DPolynomial2, TraceInner, TraceOuter, TracePreEstimation, TraceVarianceCalculation
from etrics.KDE import KDEBuilder
from etrics.quantreg import quantreg, NoDataError
from etrics.Utilities import cartesian, Timing, enum, EventHook
from scipy.stats import norm, scoreatpercentile
import numpy as np
import scipy as sp
from fractions import Fraction
from numpy.linalg import LinAlgError
import statsmodels.api as sm
import statsmodels.nonparametric as nonpar
import collections
import pickle
import logging
from scipy.stats.distributions import norm

Interpolation = enum('Linear', 'Quadratic')
Integration = enum('Density', 'Sampling', 'Degenerate')
Heterogeneity = enum('Diagonal', 'ExternalCovariates')

def gridball(self, dimensions, sizeperdim):
    eps = 0.05
    x2y2=cartesian([np.linspace(0+eps,1-eps,sizeperdim)**2]*(dimensions-1)).tolist()
    for i in x2y2: i.append(1-np.sum(i))
    x2y2=np.array(x2y2)
    return np.sqrt(x2y2[np.where(np.all(x2y2>=0, axis=1))])

class NLRQSystem:
    def __init__(self, endog, exog, endogtype, exogtype, bw, componentwise, imethod=Interpolation.Linear, trace=False, opol = 2, logger = None, boundaries = None, wavgdens = None, kernel=norm, intmethod=Integration.Sampling, empdist = None, parallel = True, heterogeneity = Heterogeneity.Diagonal):
        self._endog = endog
        self._exog = exog
        self._endogtype = endogtype
        self._exogtype = exogtype
        self.bw = bw
        self.boundaries = boundaries
        self.wavgdens = wavgdens
        self.kernel = kernel
        self.imethod = intmethod
        self.empdist = empdist
        print("Warning: Parallel set to false for debugging")
        self.parallel = heterogeneity == Heterogeneity.Diagonal
        self.heterogeneity = heterogeneity
        self._globalresid = None
        self.parlen = sum(map(lambda i: self.exog.shape[1] ** i, range(opol+1)))
        self.polynomial, self.dpolynomial = (Polynomial2, DPolynomial2) if opol == 2 else (Polynomial1, DPolynomial1)
        self.sizeperdim = 5
        self.eps = 0.1
        self.componentwise = componentwise
        self.interpolationmethod = imethod
        self.trace = trace
        self.results = None
        self.variances = None
        self.nlrqmodels = {}
        self.PreEstimation = EventHook()
        self.x1_cf = None
        if logger is None:
            self.logger = logging.getLogger("nlrq")
        else:
            self.logger = logger

    def SetExternal(self, external, externaltype):
        self.external = external
        self._externaltype = externaltype

    @property
    def exog(self):
        return self._exog

    @property
    def endog(self):
        return self._endog

    @property
    def tau(self):
        return self._tau

    @property
    def residuals(self):
        resid = np.empty(self.endog.shape)
        if self.results is not None:
            for i in range(resid.shape[0]):
                resid[i,:] = self.endog[i,:] - self.predict(self.exog[i,:], fix=False, ignorenans = True)["f"]
        else:
            self.logger.error("Fitgrid first!")

        return resid

    @property
    def taus(self):
        if self.endog.shape[1] == 1:
            def lcsmoother(y0x0):
                design = (self.exog - np.mean(self.exog, axis=0)) / np.std(self.exog, axis=0)
                (n, d) = self.exog.shape
                m = nonpar.kernel_regression.KernelReg((self.endog < y0x0[0]).astype(float), design, self._exogtype, reg_type='lc', bw=d*[0.2*(n**(-1/(4+d)))])
                f, df =  m.fit(data_predict = (np.array(y0x0[1:]) - np.mean(self.exog, axis=0))/ np.std(self.exog, axis=0))
                return f[0]
            yx0s = np.concatenate([self.endog, self.exog], axis = 1).tolist()
            taus = np.matrix(list(map(lcsmoother, yx0s))).T
            return taus
        else:
            self.logger.error("Works for Univariate Y only, implement vector QR if needed!")
            return self.residuals

    @property
    def Results(self):
        if self.results is None:
            self.logger.error("Fitgrid first!")
        return self.results

    @property
    def Variances(self):
        if self.variances is None:
            self.logger.error("Fitgrid first!")
        return self.variances

    @property
    def Covariances(self):
        if self.variances is None:
            self.logger.error("Fitgrid first!")
        return self.covariances

    def load(self, dir, name, tau, caller):
        with open("{0}/Estimates.{1}.{2}.{3}.bin".format(dir, caller, name, int(tau*100)), "rb") as f:
            self.__dict__ = pickle.load(f)
            self.__dict__["logger"] = logging.getLogger("nlrq")

    def save(self, dir, name, tau, caller):
        with open("{0}/Estimates.{1}.{2}.{3}.bin".format(dir, caller, name, int(tau*100)), "wb") as f:
            tosync = self.__dict__.copy()
            tosync.pop("logger", None)
            pickle.dump(tosync, f)
            #pickle.dump(self.__dict__, f)


    def predict(self, x0, fix = True, includevar = False, ignorenans = False):
        if self.results is None or (includevar and self.variances is None):
            self.logger.error("Fitgrid first!")
        elif fix and False in [x0[dim] == val for dim,val in self.fixdim.items()]:
            raise Exception("cannot predict, at least one dimension of x0 = {0} was fixed to another value. Fixed values: {1}".format(x0, self.fixdim))
        elif includestderr:
            return self.results.predict(x0, ignorenans = ignorenans), self.variances.predict(x0, ignorenans = ignorenans)
        else:
            return self.results.predict(x0, ignorenans = ignorenans)

    def PrepareFit(self, tau, nvectors = 5, interiorpoint = True, parallel = True):
        #self.resid = np.zeros(self.exog.shape[0]*L).reshape(self.exog.shape[0], L)
        M = self.endog.shape[1]
        self._tau = tau
        Omega = gridball(M, nvectors) if not self.componentwise else np.identity(M)
        L = Omega.shape[0]
        for i in range(L):
            y = np.dot(self.endog, Omega[i,:].T).reshape(self.endog.shape[0])
            if interiorpoint:
                self.nlrqmodels[i] = quantreg(y, self.exog, tau=tau)
            else:
                self.nlrqmodels[i] = NLRQ(y, self.exog, endogtype=self._endogtype[i], exogtype=self._exogtype, \
                    tau=self, f=self.polynomial, Df=self.dpolynomial, parlen=self.parlen)
            if self.trace and not parallel:
                # pickler cannot deal with lambda expression
                self.nlrqmodels[i].PreEstimation += self.trcpre
                self.nlrqmodels[i].PostVarianceCalculation += self.trcvar

    def trcpre(info):
        TracePreEstimation(self.logger, info)
    def trcvar(info):
        TraceVarianceCalculation(self.logger, info)

    def fit(self, x0, ignorenodataerrors = True, tau = None):
        if self.heterogeneity == Heterogeneity.Diagonal:
            return self.fit_(x0, ignorenodataerrors, tau)
        elif self.heterogeneity == Heterogeneity.ExternalCovariates:
            return self.fit_external_(x0, ignorenodataerrors, tau)

    def fit_(self, x0, ignorenodataerrors, tau):
        M = self.endog.shape[1]
        Omega = gridball(M, 5) if not self.componentwise else np.identity(M)
        L = Omega.shape[0]
        K = self.exog.shape[1]
        Lambda, mu0, mu1 = {}, {}, {}
        Lambda["par"] = np.zeros(L*(K+1)).reshape(L, K+1)
        Lambda["var"] = np.zeros(L*(K+1)).reshape(L, K+1)
        Lambda["cov"] = np.zeros(L*(K+1)).reshape(L, K+1)
        logger = logging.getLogger('collective2stage')

        for i in range(L):
            #with Timing("Fit", True):
            try:
                self.nlrqmodels[i].set_x1_cf(self.x1_cf)
                nlrqresults = self.nlrqmodels[i].fit(x0 = x0, kernel = self.kernel, bw = self.bw, tau = self._tau if tau == None else tau)
            except NoDataError as e:
                if ignorenodataerrors and self.trace:
                    logger.warning("NoDataError ignored: " + str(e))
                    Lambda["par"][i,:] = np.zeros(K+1)
                    Lambda["var"][i,:] = np.ones(K+1)
                    Lambda["cov"][i,:] = np.ones(K+1)
                else:
                    raise(e)
            else:
                Lambda["par"][i,:] = nlrqresults.params[0:K+1]
                Lambda["var"][i,:] = np.diagonal(nlrqresults.varcov)[0:K+1]
                Lambda["cov"][i,:] = np.diagonal(nlrqresults.varcov_cf)[0:K+1]
                #self.resid[:,i] = nlrqresults.resid.T

        for w in Lambda.keys():
            b = Lambda[w].T.reshape(L*(K+1)) # b=vec(Lambda)
            A = np.kron(np.identity(K+1), Omega)
            q,r = sp.linalg.qr(A, mode='economic')
            x = np.dot(np.linalg.inv(r), np.dot(q.T, b)) # vec(mu) with mu = [mu0, mu1]
            mu = x.reshape(K+1, M).T
            mu0[w], mu1[w] = mu[:,0], np.delete(mu, 0, 1)

        # Derivative form:
        # y1 dy1/dx1 dy1/dx2, dy1/dx3
        # y2 dy2/dx1 dy2/dx2, dy2/dx3
        if(self.trace):
            for f,v in [(mu0["par"], mu0["var"]),(mu1["par"],mu1["var"])]:
                stars = lambda dy, sdy: "*"*(sum(np.abs(dy/sdy) > np.array(list(map(norm.ppf, [.95,.975,.995])))))
                logger.debug("  ".join(["{:.3f} ({:.3f}) [{}]".format(dy, np.sqrt(vardy), stars(dy,np.sqrt(vardy))) for dy, vardy in zip(f.flatten(), v.flatten())]))

        return [mu0["par"], mu1["par"], mu0["var"], mu1["var"], mu0["cov"], mu1["cov"]]

    def fit_external_(self, x0, ignorenodataerrors, tau):
        from scipy.optimize import differential_evolution
        self.external0 = np.percentile(self.external, 100*( self._tau if tau == None else tau), axis = 0)
        x0z0 = np.hstack([x0, self.external0])
        b = KDEBuilder(Y = self.endog, X = self.exog, Z = self.external)
        g = b.build("YXZ|XZ_Z")
        def obj(y0):
            y0x0z0 = np.hstack([y0, x0z0])
            return np.dot(g(y0x0z0, x0z0), g(y0x0z0, x0z0))

        boundss = list(zip(list(np.percentile(self.endog, 10, axis = 0)), list(np.percentile(self.endog, 90, axis = 0))))
        res = differential_evolution(obj, bounds = boundss, polish = True, disp = False)
        f_gen = res.x
        if False:
            y1y2 = cartesian([np.linspace(*bounds, 50) for bounds in boundss])
            f_grid = y1y2[np.argmin(list(map(obj, y1y2)))]
            print("f({}) = {} =?= {}".format(x0, f_gen, f_grid))

            import matplotlib.pyplot as plt
            from matplotlib import cm
            gy = lambda i: (lambda y0: g(np.hstack([y0, x0z0]), x0z0)[i])
            fig= plt.figure(figsize=(6,12))
            objs = list(map(obj, y1y2))
            obj1 = list(map(gy(0), y1y2))
            obj2 = list(map(gy(1), y1y2))
            y1, y2 = np.unique(y1y2[:,0]), np.unique(y1y2[:,1])
            b_ = lambda o: np.max(np.abs([np.min(o), np.max(o)]))
            ax1 = fig.add_subplot(311)
            ax1.contourf(y1, y2, np.array(objs).reshape(50,50).T, levels = 50, cmap = cm.binary)
            ax2 = fig.add_subplot(312)
            ax2.contourf(y1, y2, np.array(obj1).reshape(50,50).T, levels = 50, cmap = cm.RdBu, vmin = -b_(obj1), vmax = b_(obj1))
            ax3 = fig.add_subplot(313)
            ax3.contourf(y1, y2, np.array(obj2).reshape(50,50).T, levels = 50, cmap = cm.RdBu, vmin = -b_(obj2), vmax = b_(obj2))
            fig.savefig("/tmp/optim.pdf")
            from IPython import embed; embed()

        #df = np.ones(self.endog.shape[1]*self.exog.shape[1]) # TODO
        df = np.kron(f_gen**2, 1/np.array(x0)) # eps = df * x / f, so letting df = f^2 / x will give us f instead of eps
        # f, df, se(f), se(df), cov(f(x0), f(x1)), cov(df(x0), df(x1)), last two should be 0 asymptotically
        return f_gen, df, f_gen * 0.0, df * 0.0, f_gen * 0.0, df * 0.0


    def fitgrid(self, tau, sizeperdim, M, fixdim = {}, unwrap_self_parallel_fitter = None, x1_cf = None):
        self.sizeperdim = sizeperdim
        x0 = np.empty(self.exog.shape[1])
        allgrididcs = [i for i in range(self.exog.shape[1]) if i not in fixdim]
        dimy = self.endog.shape[1]
        self.x1_cf = x1_cf
        if self.boundaries is None:
            xmins, xmaxs = np.min(self.exog, axis=0), np.max(self.exog, axis=0)
        else:
            xmins = [np.percentile(x, b*100) for x,b in zip(self.exog.T, self.boundaries)] # 0, 100 are min, max respectively
            xmaxs = [np.percentile(x, (1.0-b)*100) for x,b in zip(self.exog.T, self.boundaries)]
        grid = []

        if self.imethod == Integration.Density or self.wavgdens == None:
            for i in allgrididcs:
                grid.append(np.linspace(xmins[i], xmaxs[i], sizeperdim))
            if len(grid) > 0:
                cgrid = cartesian(grid)
            else:
                cgrid = np.array([])
            M_ = sizeperdim * self.wavgdens.k_vars if self.wavgdens is not None else 0
        elif self.imethod == Integration.Sampling or self.imethod == Integration.Degenerate:
            for i in allgrididcs[:-self.wavgdens.k_vars]:
                grid.append(np.linspace(xmins[i], xmaxs[i], sizeperdim))
            if len(grid) == 0:
                cgrid = self.empdist.sample(M)
            else:
                cgrid = np.hstack([np.kron(cartesian(grid), np.ones((M,1))), self.empdist.sample(M*len(cartesian(grid)))])
            M_ = M

        j = 0
        #mu, sigma = np.mean(self.exog, axis=0), np.std(self.exog, axis=0)
        #kernel.SetSigmas(sigma)
        self.kernel.SetSigmas(np.ones(self.exog.shape[1]))
        self.PrepareFit(tau, parallel = self.parallel)
        if len(cgrid) > 0:
            finalgrid = [[v for k,v in sorted(list(zip(allgrididcs, x0r))+list(zip(list(fixdim.keys()), list(fixdim.values()))))] for x0r in cgrid]
        else:
            finalgrid = [list(fixdim.values())]
        #print(finalgrid)
        self.results = NLRQResult(len(finalgrid), self.exog.shape[1], dimy, self.interpolationmethod, fixdim, isvariance = False) # 2nd was len(allgrididcs)
        self.variances = NLRQResult(len(finalgrid), self.exog.shape[1], dimy, self.interpolationmethod, fixdim, isvariance = True)
        self.covariances = NLRQResult(len(finalgrid), self.exog.shape[1], dimy, self.interpolationmethod, fixdim, isvariance = True)
        if not self.parallel:
            for x0 in finalgrid:
                with Timing("nlrqsystem({0}) at {1}: grid {2}^{3}".format(tau, x0, sizeperdim, len(allgrididcs)), trc = self.trace, logger = self.logger):
                    self.PreEstimation.Fire(Fraction(j, len(finalgrid)))
                    fct, grad, fctvar, gradvar, fctcov, gradcov = self.fit(x0)
                    #self.results.Add(np.array(x0)[allgrididcs], fct, np.delete(grad, list(fixdim.keys()), 1).flatten(0))
                    #self.variances.Add(np.array(x0)[allgrididcs], fctvar, np.delete(gradvar, list(fixdim.keys()), 1).flatten(0))
                    self.results.Add(np.array(x0), fct, grad.flatten())
                    self.variances.Add(np.array(x0), fctvar, gradvar.flatten())
                    self.covariances.Add(np.array(x0), fctcov, gradcov.flatten())
                j += 1
        else:
            from multiprocessing import Pool
            from os import cpu_count
            import copy
            stmp = copy.copy(self)
            nores = [stmp.__dict__.pop(k) for k in self.__dict__ if k not in ["_endog", "_exog", "trace", "componentwise", "bw", "kernel", "nlrqmodels", "_tau", "x1_cf", "heterogeneity"]]
            with Timing("nlrqsystem.par({0}) at {1}: grid {2}^{3}".format(tau, x0, sizeperdim, len(allgrididcs)), trc = self.trace, logger = self.logger):
                with Pool(processes = cpu_count() - 1) as p:
                    fitres = p.map(unwrap_self_parallel_fitter, zip([stmp]*len(finalgrid), finalgrid))
            for x0, res in zip(finalgrid, fitres):
                fct, grad, fctvar, gradvar, fctcov, gradcov = res
                #self.results.Add(np.array(x0)[allgrididcs], fct, np.delete(grad, list(fixdim.keys()), 1).flatten(0))
                #self.variances.Add(np.array(x0)[allgrididcs], fctvar, np.delete(gradvar, list(fixdim.keys()), 1).flatten(0))
                self.results.Add(np.array(x0), fct, grad.flatten())
                self.variances.Add(np.array(x0), fctvar, gradvar.flatten())
                self.covariances.Add(np.array(x0), fctcov, gradcov.flatten())

        # self.nlrqmodels = None # better way to free it? GC should get them...
        self.fixdim = fixdim
        #self.results.populatetree(self.wavgdens, allgrididcs, self.sizeperdim, M_)
        #self.variances.populatetree(self.wavgdens, allgrididcs, self.sizeperdim, M_)
        self.results.populatetree(self.wavgdens, range(self.exog.shape[1]), self.sizeperdim, M_)
        self.variances.populatetree(self.wavgdens, range(self.exog.shape[1]), self.sizeperdim, M_)
        self.covariances.populatetree(self.wavgdens, range(self.exog.shape[1]), self.sizeperdim, M_)


class NLRQResult:
    def __init__(self, N, dimX, dimY, interpolationmethod, fixdim, isvariance = False):
        self._resultmatrix = np.empty((N, dimX+dimY+dimX*dimY))
        self.dimX = dimX
        self.dimY = dimY
        self.actindex = 0
        self.resulttree = {}
        self.errordim = None
        self.interpolationmethod = interpolationmethod
        self.fixdim = fixdim
        self.isvariance = isvariance

    def Add(self, x, y, Dy):
        self._resultmatrix[self.actindex,:] = np.concatenate([x, y, Dy])
        self.actindex += 1

    @property
    def X(self):
        return self._resultmatrix[:,:(self.dimX-self.errordim)]

    @property
    def Y(self):
        return self._resultmatrix[:,(self.dimX-self.errordim):-(self.dimX*self.dimY)]

    @property
    def DY(self):
        return self._resultmatrix[:,-(self.dimX*self.dimY):]

    def partialDY(self, idx):
        return self.DY[:,(self.dimX*idx):(self.dimX*(idx+1))]

    @property
    def Everything(self):
        return self._resultmatrix

    @property
    def EverythingEps(self):
        return self._epsmatrix

    @property
    def Epsilon(self):
        return self._epsmatrix[:,(self.dimX-self.errordim):]

    @property
    def EverythingVarE(self):
        return self._resultmatrixVarE

    def _populatetree(self, t, z, zeps, sizey, sizedy):
        if z.shape[1] > sizey+sizedy:
            for v in np.unique(np.asarray(z[:,0]).flatten()):
                t[v] = {}
                self._populatetree(t[v], z[np.asarray(z[:,0]).flatten()==v][:,1:], zeps[np.asarray(zeps[:,0]).flatten()==v][:,1:], sizey, sizedy)
        else:
            t['f'] = z[:,:sizey]
            t['df'] = z[:,-sizedy:]
            t['eps'] = zeps

    def populatetree(self, wavgdens, allgrididcs, sizeperdim, M):
        #self.results.populatetree(self.results.integrate(wavgdens, allgrididcs, self.sizeperdim), \
        #    self.endog.shape[1], len(allgrididcs)*self.endog.shape[1])
        self.integrate(wavgdens, allgrididcs, sizeperdim, M)
        self._populatetree(self.resulttree, self.Everything, self.EverythingEps, self.dimY, self.dimX*self.dimY)
        self.grididcs = allgrididcs if wavgdens is None else allgrididcs[:-wavgdens.k_vars]

    def integrate(self, wavgdens, pgrididcs, sizeperdim, M):
        self.errordim = wavgdens.k_vars if wavgdens is not None else 0
        x0as = self.Everything[:,:len(pgrididcs)]
        ys = self.Everything[:,len(pgrididcs):(len(pgrididcs)+self.dimY)]
        dys = self.Everything[:,(len(pgrididcs)+self.dimY):]
        if not self.isvariance:
            A = np.kron(1/ys, np.ones(x0as.shape[1]).T)
            B = np.kron(np.ones(self.dimY).T, x0as)
            eps = np.multiply(np.multiply(A, B), dys)
        else: # TODO change using derivative (delta rule)
            #from IPython import embed; embed()
            #eps = np.multiply(np.multiply(A, B), dys)
            eps = dys*0.0
        if wavgdens is not None:
            xdim = len(pgrididcs) - self.errordim
            #M = sizeperdim ** self.errordim
            J = self.dimY * (1 + len(pgrididcs))
            L = sizeperdim ** xdim
            weights = np.array([w for w in map(wavgdens.pdf, self.Everything[:,:len(pgrididcs)][:,-self.errordim:])])
            #print(self.errordim, xdim, M, J, L, sizeperdim, pgrididcs, weights.shape)
            #weights = np.ones(weights.shape)
            weights /= np.sum(weights)/L
            reducedxidcs = np.array(range(L)) * M
            x0s = x0as[reducedxidcs,:-self.errordim]
            eps = np.multiply(eps, np.matrix(weights).T).reshape(L, self.dimY * len(pgrididcs) * M)
            eps = np.dot(eps, np.kron(np.ones((M, 1)), np.identity(self.dimY * len(pgrididcs))))
            ydys = self.Everything[:,len(pgrididcs):]
            ydys = np.multiply(ydys, np.matrix(weights).T).reshape(L, J * M)
            ydys = np.dot(ydys, np.kron(np.ones((M, 1)), np.identity(J)))
            restmp = ys if xdim == 0 else np.hstack([x0s, ydys])
            epstmp = eps if xdim == 0 else np.hstack([x0s, eps])
            ys2 = np.multiply(np.power(self.Everything[:,len(pgrididcs):], 2), np.matrix(weights).T).reshape(L, J * M)
            ys2 = np.dot(ys2, np.kron(np.ones((M, 1)), np.identity(J)))
            restmpvarx = ys2 if xdim == 0 else np.hstack([x0s, ys2 - np.power(ydys, 2)])
            #print(restmpvarx)
            #print(np.hstack([self.Everything[0:M,0:6], 100*np.matrix(weights[0:M]).T]), restmp[0,0:xdim+self.dimY])
            #print(restmp[:,0:len(pgrididcs)+self.dimY])
        else:
            restmp = self.Everything
            restmpvarx = np.zeros(self.Everything.shape)
            epstmp = eps if len(pgrididcs) == 0 else np.hstack([x0as, eps])

        self._resultmatrix = restmp
        self._epsmatrix = epstmp
        self._resultmatrixVarE = restmpvarx

    def interpolate(self, node, x0):
        if len(x0) > 0:
            snode = np.array(sorted(node))
            idx = snode.searchsorted(x0[0])
            if x0[0] >= snode[0] and x0[0] <= snode[-1]:
                lx, ux = snode[idx-1], snode[idx]
                ly, ldy, leps = list(self.interpolate(node[lx], x0[1:]))
                uy, udy, ueps = list(self.interpolate(node[ux], x0[1:]))

                if self.interpolationmethod == Interpolation.Quadratic and len(snode) >= 3:
                    islower = (x0[0]-lx < ux-x0[0] and idx-2 >= 0) or (idx+1 >= len(snode))
                    ex = snode[idx-2] if islower else snode[idx+1]
                    ey, edy, eeps = self.interpolate(node[ex], x0[1:])
                    for [y0, y1, y2] in [[ly, uy, ey], [ldy, udy, edy], [leps, ueps, eeps]]:
                        yield y2*(x0[0]-lx)*(x0[0]-ux)/((ex-lx)*(ex-ux))+\
                            y0*(x0[0]-ex)*(x0[0]-ux)/((lx-ex)*(lx-ux))+\
                            y1*(x0[0]-ex)*(x0[0]-lx)/((ux-ex)*(ux-lx))
                else:
                    for [y0, y1] in [[ly, uy], [ldy, udy], [leps, ueps]]:
                        yield y0+(x0[0]-lx)*(y1-y0)/(ux-lx)
            else: # outside the estimated grid, too close to the boundary, only if toboundary=False
                yield np.nan
                yield np.nan
                yield np.nan
        else:
            yield node['f']
            yield node['df']
            yield node['eps']

    # fixdim should not have any business here (it's treated now just like any other x0)
    def predictOld(self, x0, ignorenans = False):
        if len(x0) < len(self.fixdim) + len(self.grididcs):
            fullx0 = np.zeros(len(self.fixdim) + len(self.grididcs))
            fullx0[list(self.fixdim.keys())] = list(self.fixdim.values())
            if len(x0) > 0:
                fullx0[self.grididcs] = x0
        else:
            fullx0 = x0

        f, df, eps = [np.asarray(val).flatten() for val in self.interpolate(self.resulttree, np.array(fullx0)[self.grididcs])]
        if np.any(np.isnan(f)) and not ignorenans:
            fullmin_g = np.zeros(len(self.fixdim) + len(self.grididcs))
            fullmin_g[list(self.fixdim.keys())] = list(self.fixdim.values())
            fullmin_g[self.grididcs] = np.min(self.X, axis=0)
            fullmax_g = np.zeros(len(self.fixdim) + len(self.grididcs))
            fullmax_g[list(self.fixdim.keys())] = list(self.fixdim.values())
            fullmax_g[self.grididcs] = np.max(self.X, axis=0)

            raise Exception("Prediction failed [ignorenans={}] x = {} f = {} df = {} eps = {} min_g = {} max_g = {}".format(ignorenans, fullx0, f, df, eps, fullmin_g, fullmax_g))
        return {"f":f, "df":df, "eps":eps}

    def predict(self, x0):
        if self.Everything.shape[0] > 1:
            f, df, eps = [np.asarray(val).flatten() for val in self.interpolate(self.resulttree, np.array(x0))]
        elif np.all(self.X == x0):
            f, df, eps = np.asarray(self.Y).flatten(), np.asarray(self.DY).flatten(), np.asarray(self.Epsilon).flatten()
        else:
            raise Exception("Prediction failed x = {} f = {} df = {} eps = {}".format(x0, f, df, eps))

        return {"f":f, "df":df, "eps":eps}

    def predictF(self, x0):
        return self.predict(x0)["f"]

    def predictDF(self, x0):
        # Form: dy1/dx1 [dy1/dx2] dy1/dx3 dy1/da1 dy1/da2 dy2/dx1 [dy2/dx2] dy2/dx3 dy2/da1 dy2/da2
        # [] for fixed second dimension fixdim = [1]
        return self.predict(x0)["df"]

    def predictEps(self, x0):
        # Form: dy1/dx1*x1/y1 dy1/dx2*x2/y1 dy1/da1*a1/y1 dy2/dx1*x1/y2 dy2/dx2*x2/y2 dy2/da1*a1/y2
        return self.predict(x0)["eps"]


# DEPRECATED
def UCIConstant(alpha, bw, C):
    return (-2 * np.log(bw)) * ((1 + (-2 * np.log(bw))) ** -1) * (-np.log(-np.log(1-alpha)/2) + np.log((C ** (1/2)) / 2*np.pi) )

# DEPRECATED
def UniformCI(x0, tau, alpha, kernel, N, bw, Q, exogdens, endogdens, exogdenshat, endogdenshat, opol):
    # takes about 0.67 sec for public good (*1000 evaluations = 11 minutes)
    # use f(x0) to calculate true quantile
    # combine the two to get a feasible sparsity
    # use information about kernel to construct variance and L(lambda) bound
    #print(list(zip(exogdens, x0)))
    #print(x0, Q(x0), endogdens(Q(x0)), np.product([d(v) for d,v in zip(exogdens, x0)]))
    useestimated = True
    if useestimated:
        fx = exogdenshat(x0)
        sparsity = 1/np.array([d(y, x0).item() for d,y in zip(endogdenshat, Q(x0))])
        #sparsity /= 60 # to trim to true one which is .75
    else:
        fx = np.product([d(v) for d,v in zip(exogdens, x0)])
        sparsity = 1/endogdens(Q(x0))

    Np, Npinv, Tp, Qp = kernel.Np, kernel.Npinv, kernel.Tp, kernel.Qp
    #Npinv = np.linalg.inv(Np)
    C_0 = (Npinv * Qp * Npinv)[1,1]/(Npinv * Tp * Npinv)[1,1]
    #L_0 = UCIConstant(alpha, bw, C_0)
    L_0 = norm.ppf(1-alpha/2)
    var = tau * (1-tau) * (Npinv * Tp * Npinv)[0,0] * (sparsity ** 2) / (fx * N * bw) # from nh^-.5 from L_0
    altvar = tau * (1-tau) * Tp[0,0] * (sparsity**2) / (fx * N * bw) # h ** len(x0) for multivariate??
    uniformbounds = np.vstack([Q(x0) - L_0*np.sqrt(var), Q(x0) + L_0*np.sqrt(var)]).T.tolist()

    return uniformbounds


# DEPRECEATED
    def fit_external_old(self, x0, ignorenodataerrors, tau):
        from statsmodels.nonparametric import kernel_density
        from scipy.optimize import differential_evolution
        self.external0 = np.percentile(self.external, 100*( self._tau if tau == None else tau), axis = 0)
        #self._externaltype = 'oo' # TODO: uncomment
        x0z0 = np.hstack([x0, self.external0])
        def obj(y0):
            y0x0z0 = np.hstack([y0, x0z0])
            num = kernel_density.KDEMultivariate(data = np.hstack([self.endog, self.exog, self.external]), var_type = self._endogtype+self._exogtype+self._externaltype, bw='normal_reference')
            den = kernel_density.KDEMultivariate(data = np.hstack([self.exog, self.external]), var_type = self._exogtype+self._externaltype, bw='normal_reference')
            return -num.pdf(y0x0z0)/den.pdf(x0z0)

        def obj_wo(y0):
            y0x0z0 = np.hstack([y0, x0[:-2], self.external0])
            num = kernel_density.KDEMultivariate(data = np.hstack([self.endog, self.exog[:,:-2], self.external]), var_type = self._endogtype+self._exogtype[:-2]+self._externaltype, bw='normal_reference')
            den = kernel_density.KDEMultivariate(data = np.hstack([self.exog[:,:-2], self.external]), var_type = self._exogtype[:-2]+self._externaltype, bw='normal_reference')
            return -num.pdf(y0x0z0)/den.pdf(np.hstack([x0[:-2], self.external0]))

        boundss = list(zip(list(np.min(self.endog, axis = 0)), list(np.max(self.endog, axis = 0))))
        res = differential_evolution(obj_wo, bounds = boundss, mutation = 1.9, strategy = 'rand1exp', polish = True, disp = False)
        f_gen = res.x
        print("f({}) = {}".format(x0, f_gen))

        df = np.ones(self.endog.shape[1]*self.exog.shape[1]) # TODO

        # f, df, se(f), se(df), cov(f(x0), f(x1)), cov(df(x0), df(x1)), last two should be 0 asymptotically
        return f_gen, df, f_gen * 0.0, df * 0.0, f_gen * 0.0, df * 0.0


