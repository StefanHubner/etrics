from etrics.NLRQ import NLRQ, Polynomial1, DPolynomial1, Polynomial2, DPolynomial2, TraceInner, TraceOuter, TracePreEstimation, TraceVarianceCalculation
from etrics.quantreg import quantreg, NoDataError
from etrics.Utilities import cartesian, Timing, enum, EventHook
from scipy.stats import norm, scoreatpercentile
import numpy as np
import scipy as sp
from fractions import Fraction
from numpy.linalg import LinAlgError
import statsmodels.api as sm
import collections
import pickle
import logging
from scipy.stats.distributions import norm

Interpolation = enum('Linear', 'Quadratic')
Integration = enum('Density', 'Sampling')

def gridball(self, dimensions, sizeperdim):
	eps = 0.05
	x2y2=cartesian([np.linspace(0+eps,1-eps,sizeperdim)**2]*(dimensions-1)).tolist()
	for i in x2y2: i.append(1-np.sum(i))
	x2y2=np.array(x2y2)
	return np.sqrt(x2y2[np.where(np.all(x2y2>=0, axis=1))])

class NLRQSystem:
	def __init__(self, endog, exog, endogtype, exogtype, tau, componentwise, imethod=Interpolation.Linear, trace=False, opol = 2, logger = None):
		self._endog = endog
		self._exog = exog
		self._endogtype = endogtype
		self._exogtype = exogtype
		self._tau = tau
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
		if logger is None:
			self.logger = logging.getLogger("nlrq")
		else:
			self.logger = logger
	
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
	def Results(self):
		if self.results is None:
			self.logger.error("Fitgrid first!")
		return self.results
	
	@property
	def StdErrors(self):
		if self.variances is None:
			self.logger.error("Fitgrid first!")
		return self.variances

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

	
	def predict(self, x0, fix = True, includestderr = False, ignorenans = False):
		if self.results is None or (includestderr and self.variances is None):
			self.logger.error("Fitgrid first!")
		elif fix and False in [x0[dim] == val for dim,val in self.fixdim.items()]:
			raise Exception("cannot predict, at least one dimension of x0 = {0} was fixed to another value. Fixed values: {1}".format(x0, self.fixdim))
		elif includestderr:
			return self.results.predict(x0, ignorenans = ignorenans), self.variances.predict(x0, ignorenans = ignorenans)
		else:
			return self.results.predict(x0, ignorenans = ignorenans)
	
	def PrepareFit(self, nvectors = 5, interiorpoint = True, parallel = True):
		#self.resid = np.zeros(self.exog.shape[0]*L).reshape(self.exog.shape[0], L)
		M = self.endog.shape[1]
		Omega = gridball(M, nvectors) if not self.componentwise else np.identity(M)
		L = Omega.shape[0]
		for i in range(L):
			y = np.dot(self.endog, Omega[i,:].T).reshape(self.endog.shape[0])
			if interiorpoint:
				self.nlrqmodels[i] = quantreg(y, self.exog, tau=self.tau)
			else:
				self.nlrqmodels[i] = NLRQ(y, self.exog, endogtype=self._endogtype[i], exogtype=self._exogtype, \
					tau=self.tau, f=self.polynomial, Df=self.dpolynomial, parlen=self.parlen)
			if self.trace and not parallel:
				# pickler cannot deal with lambda expression
				def trcpre(info): TracePreEstimation(self.logger, info) 
				def trcvar(info): TraceVarianceCalculation(self.logger, info)
				self.nlrqmodels[i].PreEstimation += trcpre
				self.nlrqmodels[i].PostVarianceCalculation += trcvar 

	def fit(self, x0, nvectors=5, ignorenodataerrors=True):
		M = self.endog.shape[1]
		Omega = gridball(M, nvectors) if not self.componentwise else np.identity(M)
		L = Omega.shape[0]
		K = self.exog.shape[1]
		Lambda, mu0, mu1 = {}, {}, {}
		Lambda["par"] = np.zeros(L*(K+1)).reshape(L, K+1)
		Lambda["var"] = np.zeros(L*(K+1)).reshape(L, K+1)
		logger = logging.getLogger('collective2stage')

		for i in range(L):
			#with Timing("Fit", True):
			try:
				nlrqresults = self.nlrqmodels[i].fit(x0 = x0, kernel = self.kernel, dist = self.exog - x0, bw = self.bw) 
			except NoDataError as e:
				if ignorenodataerrors and self.trace:
					logger.warning("NoDataError ignored: " + str(e))
					Lambda["par"][i,:] = np.zeros(K+1) 
					Lambda["var"][i,:] = np.ones(K+1)
				else:
					raise(e) 
			else:	
				Lambda["par"][i,:] = nlrqresults.params[0:K+1]
				Lambda["var"][i,:] = np.diagonal(nlrqresults.varcov)[0:K+1]
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

		return [mu0["par"], mu1["par"], mu0["var"], mu1["var"]] 

	def fitgrid(self, h, sizeperdim, M, fixdim = {}, boundaries = None, wavgdens = None, kernel=norm, imethod=Integration.Sampling, empdist = None, parallel = True, unwrap_self_parallel_fitter = None): 
		self.sizeperdim = sizeperdim
		self.kernel = kernel
		self.bw = h 
		x0 = np.empty(self.exog.shape[1])
		allgrididcs = [i for i in range(self.exog.shape[1]) if i not in fixdim]
		dimy = self.endog.shape[1]
		if boundaries is None: 
			xmins, xmaxs = np.min(self.exog, axis=0), np.max(self.exog, axis=0)
		else:
			xmins = [np.percentile(x, b*100) for x,b in zip(self.exog.T, boundaries)] # 0, 100 are min, max respectively
			xmaxs = [np.percentile(x, (1.0-b)*100) for x,b in zip(self.exog.T, boundaries)] 
		grid = []

		if imethod == Integration.Density or wavgdens == None:
			for i in allgrididcs:
				grid.append(np.linspace(xmins[i], xmaxs[i], sizeperdim))
			cgrid = cartesian(grid)
			M_ = sizeperdim * wavgdens.k_vars if wavgdens is not None else 0
		elif imethod == Integration.Sampling:
			for i in allgrididcs[:-wavgdens.k_vars]:
				grid.append(np.linspace(xmins[i], xmaxs[i], sizeperdim))
			if len(grid) == 0:	
				cgrid = empdist.sample(M)
			else:
				cgrid = np.hstack([np.kron(cartesian(grid), np.ones((M,1))), empdist.sample(M*len(cartesian(grid)))])
			M_ = M	

		self.results = NLRQResult(cgrid.shape[0], len(allgrididcs), dimy, self.interpolationmethod, fixdim)
		self.variances = NLRQResult(cgrid.shape[0], len(allgrididcs), dimy, self.interpolationmethod, fixdim)
		j = 0
		mu, sigma = np.mean(self.exog, axis=0), np.std(self.exog, axis=0)
		#kernel.SetSigmas(sigma) 
		kernel.SetSigmas(np.ones(self.exog.shape[1])) 
		self.PrepareFit(parallel = parallel)
		finalgrid = [[v for k,v in sorted(list(zip(allgrididcs, x0r))+list(zip(list(fixdim.keys()), list(fixdim.values()))))] for x0r in cgrid]
		if not parallel:
			for x0 in finalgrid:
				with Timing("nlrqsystem({0}) at {1}: grid {2}^{3}".format(self.tau, x0, sizeperdim, len(allgrididcs)), trc = self.trace, logger = self.logger): 
					self.PreEstimation.Fire(Fraction(j, len(cgrid)))
					fct, grad, fctse, gradse = self.fit(x0) 
					self.results.Add(np.array(x0)[allgrididcs], fct, np.delete(grad, list(fixdim.keys()), 1).flatten(0))
					self.variances.Add(np.array(x0)[allgrididcs], fctse, np.delete(gradse, list(fixdim.keys()), 1).flatten(0))
				j += 1	
		else:
			from multiprocessing import Pool
			from os import cpu_count
			import copy
			stmp = copy.copy(self)
			nores = [stmp.__dict__.pop(k) for k in self.__dict__ if k not in ["_endog", "_exog", "trace", "componentwise", "bw", "kernel", "nlrqmodels"]]
			with Timing("nlrqsystem.par({0}) at {1}: grid {2}^{3}".format(self.tau, x0, sizeperdim, len(allgrididcs)), trc = self.trace, logger = self.logger): 
				with Pool(processes = cpu_count() - 1) as p:
					fitres = p.map(unwrap_self_parallel_fitter, zip([stmp]*len(finalgrid), finalgrid))
			for x0,res in zip(finalgrid, fitres):
				fct, grad, fctse, gradse = res 
				self.results.Add(np.array(x0)[allgrididcs], fct, np.delete(grad, list(fixdim.keys()), 1).flatten(0))
				self.variances.Add(np.array(x0)[allgrididcs], fctse, np.delete(gradse, list(fixdim.keys()), 1).flatten(0))

		self.nlrqmodels = None # better way to free it? GC should get them...
		self.fixdim = fixdim
		self.results.populatetree(wavgdens, allgrididcs, self.sizeperdim, M_)
		self.variances.populatetree(wavgdens, allgrididcs, self.sizeperdim, M_)
	
				
class NLRQResult:
	def __init__(self, N, dimX, dimY, interpolationmethod, fixdim):
		self._resultmatrix = np.empty((N, dimX+dimY+dimX*dimY))
		self.dimX = dimX
		self.dimY = dimY
		self.actindex = 0
		self.resulttree = {}
		self.errordim = None
		self.interpolationmethod = interpolationmethod
		self.fixdim = fixdim
	
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
	def EverythingVarE(self):
		return self._resultmatrixVarE

	def _populatetree(self, t, z, sizey, sizedy):
		if z.shape[1] > sizey+sizedy:
			for v in np.unique(np.asarray(z[:,0]).flatten()):
				t[v] = {}
				self._populatetree(t[v], z[np.asarray(z[:,0]).flatten()==v][:,1:], sizey, sizedy)
		else:
			t['f'] = z[:,:sizey]
			t['df'] = z[:,-sizedy:]

	def populatetree(self, wavgdens, allgrididcs, sizeperdim, M):	
		#self.results.populatetree(self.results.integrate(wavgdens, allgrididcs, self.sizeperdim), \
		#	self.endog.shape[1], len(allgrididcs)*self.endog.shape[1])
		self.integrate(wavgdens, allgrididcs, sizeperdim, M)
		self._populatetree(self.resulttree, self.Everything, self.dimY, self.dimX*self.dimY) 
		self.grididcs = allgrididcs if wavgdens is None else allgrididcs[:-wavgdens.k_vars]

	def integrate(self, wavgdens, pgrididcs, sizeperdim, M):	
		if wavgdens is not None:
			self.errordim = wavgdens.k_vars
			xdim = len(pgrididcs) - self.errordim 
			#M = sizeperdim ** self.errordim
			J = self.dimY * (1 + len(pgrididcs))
			L = sizeperdim ** xdim
			weights = np.array([w for w in map(wavgdens.pdf, self.Everything[:,:len(pgrididcs)][:,-self.errordim:])])
			#print(self.errordim, xdim, M, J, L, sizeperdim, pgrididcs, weights.shape)
			#weights = np.ones(weights.shape)
			weights /= np.sum(weights)/L
			reducedxidcs = np.array(range(L)) * M 
			x0s = self.Everything[reducedxidcs,:len(pgrididcs)][:,:-self.errordim]
			ys = np.multiply(self.Everything[:,len(pgrididcs):], np.matrix(weights).T).reshape(L, J * M)
			ys = np.dot(ys, np.kron(np.ones((M, 1)), np.identity(J)))
			restmp = ys if xdim == 0 else np.hstack([x0s, ys])
			ys2 = np.multiply(np.power(self.Everything[:,len(pgrididcs):], 2), np.matrix(weights).T).reshape(L, J * M)
			ys2 = np.dot(ys2, np.kron(np.ones((M, 1)), np.identity(J)))
			restmpvarx = ys2 if xdim == 0 else np.hstack([x0s, ys2 - np.power(ys, 2)])
			#print(restmpvarx)
			#print(np.hstack([self.Everything[0:M,0:6], 100*np.matrix(weights[0:M]).T]), restmp[0,0:xdim+self.dimY])
			#print(restmp[:,0:len(pgrididcs)+self.dimY])
		else: 
			self.errordim = 0
			restmp = self.Everything 
			restmpvarx = np.zeros(self.Everything.shape)

		self._resultmatrix = restmp
		self._resultmatrixVarE = restmpvarx
	
	def interpolate(self, node, x0):
		if len(x0) > 0:
			snode = np.array(sorted(node))
			idx = snode.searchsorted(x0[0])
			if x0[0] >= snode[0] and x0[0] <= snode[-1]:
				lx, ux = snode[idx-1], snode[idx]
				ly, ldy = list(self.interpolate(node[lx], x0[1:]))
				uy, udy = list(self.interpolate(node[ux], x0[1:]))

				if self.interpolationmethod == Interpolation.Quadratic and len(snode) >= 3:
					islower = (x0[0]-lx < ux-x0[0] and idx-2 >= 0) or (idx+1 >= len(snode)) 
					ex = snode[idx-2] if islower else snode[idx+1]
					ey, edy = self.interpolate(node[ex], x0[1:])
					for [y0, y1, y2] in [[ly, uy, ey], [ldy, udy, edy]]:
						yield y2*(x0[0]-lx)*(x0[0]-ux)/((ex-lx)*(ex-ux))+\
							y0*(x0[0]-ex)*(x0[0]-ux)/((lx-ex)*(lx-ux))+\
							y1*(x0[0]-ex)*(x0[0]-lx)/((ux-ex)*(ux-lx))
				else:	
					for [y0, y1] in [[ly, uy], [ldy, udy]]:
						yield y0+(x0[0]-lx)*(y1-y0)/(ux-lx)
			else: # outside the estimated grid, too close to the boundary, only if toboundary=False	
				yield np.nan
				yield np.nan
		else:
			yield node['f']
			yield node['df']
	
	def predict(self, x0, ignorenans = False):	
		if len(x0) < len(self.fixdim) + len(self.grididcs):
			fullx0 = np.zeros(len(self.fixdim) + len(self.grididcs))
			fullx0[list(self.fixdim.keys())] = list(self.fixdim.values())
			fullx0[self.grididcs] = x0
		else:
			fullx0 = x0

		f, df = [np.asarray(val).flatten() for val in self.interpolate(self.resulttree, np.array(fullx0)[self.grididcs])]
		if np.any(np.isnan(f)) and not ignorenans: 
			fullmin_g = np.zeros(len(self.fixdim) + len(self.grididcs))
			fullmin_g[list(self.fixdim.keys())] = list(self.fixdim.values())
			fullmin_g[self.grididcs] = np.min(self.X, axis=0) 
			fullmax_g = np.zeros(len(self.fixdim) + len(self.grididcs))
			fullmax_g[list(self.fixdim.keys())] = list(self.fixdim.values())
			fullmax_g[self.grididcs] = np.max(self.X, axis=0) 

			raise Exception("Prediction failed [ignorenans={}] x = {} f = {} df = {} min_g = {} max_g = {}".format(ignorenans, fullx0, f, df, fullmin_g, fullmax_g))
		return {"f":f, "df":df} 

	def predictF(self, x0):
		return self.predict(x0)["f"]	

	def predictDF(self, x0):
		# Form: dy1/dx1 [dy1/dx2] dy1/dx3 dy1/da1 dy1/da2 dy2/dx1 [dy2/dx2] dy2/dx3 dy2/da1 dy2/da2 
		# [] for fixed second dimension fixdim = [1]
		return self.predict(x0)["df"]	


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

