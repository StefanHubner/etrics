from etrics.NLRQ import NLRQ, Polynomial1, DPolynomial1, Polynomial2, DPolynomial2, TraceInner, TraceOuter
from etrics.Utilities import cartesian, Timing, enum, EventHook
from scipy.stats import norm, scoreatpercentile
import numpy as np
import scipy as sp
from fractions import Fraction
import statsmodels.api as sm
import collections
import pickle
from scipy.stats.distributions import norm

Interpolation = enum('Linear', 'Quadratic')

class NLRQSystem:
	def __init__(self, endog, exog, tau, componentwise, imethod=Interpolation.Linear, trace=False, opol = 2):
		self._endog = endog
		self._exog = exog
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
		self.PreEstimation = EventHook()
	
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
				resid[i,:] = self.endog[i,:] - self.predict(self.exog[i,:], fix=False)["f"]
		else:
			print("fitgrid first")

		return resid
	
	@property
	def Results(self):
		if self.results is None:
			print("fitgrid first")
		return self.results

	def load(self, dir, name):
		with open("{0}/Estimates.{1}.bin".format(dir, name), "rb") as f:
			self.__dict__ = pickle.load(f)
	
	def save(self, dir, name):
		with open("{0}/Estimates.{1}.bin".format(dir, name), "wb") as f:
			pickle.dump(self.__dict__, f)
	
	def predict(self, x0, fix = True):
		if self.results is None:
			print("fitgrid first")
		elif fix and False in [x0[dim] == val for dim,val in self.fixdim.items()]:
			raise Exception("cannot predict, at least one dimension of x0 = {0} was fixed to another value. Fixed values: {1}".format(x0, self.fixdim))
		else:
			return self.results.predict(x0)
	
	def fit(self, x0, weights):
		M = self.endog.shape[1]
		Omega = self.gridball(M, 5) if not self.componentwise else np.identity(M)
		L = Omega.shape[0]
		K = self.exog.shape[1]
		Lambda = np.zeros(L*(K+1)).reshape(L, K+1)
		self.resid = np.zeros(self.exog.shape[0]*L).reshape(self.exog.shape[0], L)

		for i in range(L):
			y = np.dot(self.endog, Omega[i,:].T).reshape(self.endog.shape[0])
			nlrqmodel = NLRQ(y, self.exog, tau=self.tau, f=self.polynomial, Df=self.dpolynomial, parlen=self.parlen)
			nlrqresults = nlrqmodel.fit(x0 = x0, weights = weights) 
			Lambda[i,:] = nlrqresults.params[0:K+1]
			self.resid[:,i] = nlrqresults.resid.T
	
		b = Lambda.T.reshape(L*(K+1)) # b=vec(Lambda)
		A = np.kron(np.identity(K+1), Omega) 
		q,r = sp.linalg.qr(A, mode='economic')
		x = np.dot(np.linalg.inv(r), np.dot(q.T, b)) # vec(mu) with mu = [mu0, mu1]
		mu = x.reshape(K+1, M).T
		mu0, mu1 = mu[:,0], np.delete(mu, 0, 1)

		return mu0, mu1 

	def fitgrid(self, h, sizeperdim, fixdim = {}, boundaries = None, wavgdens = None, kernel=norm): 
		self.sizeperdim = sizeperdim
		x0 = np.empty(self.exog.shape[1])
		allgrididcs = [i for i in range(self.exog.shape[1]) if i not in fixdim]
		dimy = self.endog.shape[1]
		if boundaries is None: 
			xmins, xmaxs = np.min(self.exog, axis=0), np.max(self.exog, axis=0)
		else:
			xmins, xmaxs = np.percentile(self.exog, boundaries[0], axis=0), np.percentile(self.exog, boundaries[1], axis=0)
		grid = []
		for i in allgrididcs:
			grid.append(np.linspace(xmins[i], xmaxs[i], sizeperdim))
		glen = len(cartesian(grid))
		cgrid = cartesian(grid)
		self.results = NLRQResult(cgrid.shape[0], len(allgrididcs), dimy, self.interpolationmethod, fixdim)
		j = 0
		for x0r in cgrid:
			x0[allgrididcs] = x0r
			if len(fixdim) > 0: x0[list(fixdim.keys())] = list(fixdim.values())
			dist = np.sum(np.abs(self.exog-x0)**2, axis=1)**.5
			weights = kernel.pdf(dist/h)
			#print(np.mean(weights > 0))
			with Timing("nlrqsystem({0}) at {1}: grid {2}^{3}".format(self.tau, x0, sizeperdim, len(allgrididcs)), self.trace):
				self.PreEstimation.Fire(Fraction(j, len(cgrid)))
				fct, grad = self.fit(x0 = x0, weights = weights) 
				self.results.Add(x0r, fct, np.delete(grad, list(fixdim.keys()), 1).flatten(0))
			j += 1	

		self.fixdim = fixdim
		#self.results.populatetree(self.results.integrate(wavgdens, allgrididcs, self.sizeperdim), \
		#	self.endog.shape[1], len(allgrididcs)*self.endog.shape[1])
		self.results.populatetree(wavgdens, allgrididcs, self.sizeperdim)
	
	def gridball(self, dimensions, sizeperdim):
		eps = 0.05
		x2y2=cartesian([np.linspace(0+eps,1-eps,sizeperdim)**2]*(dimensions-1)).tolist()
		for i in x2y2: i.append(1-np.sum(i))
		x2y2=np.array(x2y2)
		return np.sqrt(x2y2[np.where(np.all(x2y2>=0, axis=1))])
				
class NLRQResult:
	def __init__(self, N, dimX, dimY, interpolationmethod, fixdim):
		self._resultmatrix = np.empty((N, dimX+dimY+dimX*dimY))
		self.dimX = dimX
		self.dimY = dimY
		self.actindex = 0
		self.resulttree = {}
		self.interpolationmethod = interpolationmethod
		self.fixdim = fixdim
	
	def Add(self, x, y, Dy):
		self._resultmatrix[self.actindex,:] = np.concatenate([x, y, Dy])
		self.actindex += 1
	
	@property
	def X(self):
		return self._resultmatrix[:,:self.dimX]

	@property
	def Y(self):
		return self._resultmatrix[:,self.dimX:-(self.dimX*self.dimY)]

	@property
	def DY(self):
		return self._resultmatrix[:,(self.dimX+self.dimY):]

	def partialDY(self, idx): 
		return self.DY[:,(self.dimX*idx):(self.dimX*(idx+1))]

	@property
	def Everything(self):
		return self._resultmatrix

	def _populatetree(self, t, z, sizey, sizedy):
		if z.shape[1] > sizey+sizedy:
			for v in np.unique(np.asarray(z[:,0]).flatten()):
				t[v] = {}
				self._populatetree(t[v], z[np.asarray(z[:,0]).flatten()==v][:,1:], sizey, sizedy)
		else:
			t['f'] = z[:,:sizey]
			t['df'] = z[:,-sizedy:]

	def populatetree(self, wavgdens, allgrididcs, sizeperdim):		
		#self.results.populatetree(self.results.integrate(wavgdens, allgrididcs, self.sizeperdim), \
		#	self.endog.shape[1], len(allgrididcs)*self.endog.shape[1])
		self.integrate(wavgdens, allgrididcs, sizeperdim)
		self._populatetree(self.resulttree, self.Everything, self.dimY, self.dimX*self.dimY) 
		self.grididcs = allgrididcs if wavgdens is None else allgrididcs[:-wavgdens.k_vars]

	def integrate(self, wavgdens, pgrididcs, sizeperdim):	
		if wavgdens is not None:
			errordim = wavgdens.k_vars
			xdim = len(pgrididcs) - errordim
			K = sizeperdim ** errordim
			J = self.dimY * (1 + len(pgrididcs))
			L = sizeperdim ** xdim
			weights = np.array([w for w in map(wavgdens.pdf, self.Everything[:,:len(pgrididcs)][:,-errordim:])])
			weights /= np.sum(weights)/L
			reducedxidcs = np.array(range(L)) * K 
			x0s = self.Everything[reducedxidcs,:len(pgrididcs)][:,:-errordim]
			ys = np.multiply(self.Everything[:,len(pgrididcs):], np.matrix(weights).T).reshape(L, J * K)
			ys = np.dot(ys, np.kron(np.ones((K, 1)), np.identity(J)))
			restmp = ys if xdim == 0 else np.hstack([x0s, ys])
		else: 
			restmp = self.Everything 

		self._resultmatrix = restmp
	
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
	
	def predict(self, x0):	
		if len(x0) < len(self.fixdim) + len(self.grididcs):
			fullx0 = np.zeros(len(self.fixdim) + len(self.grididcs))
			fullx0[list(self.fixdim.keys())] = list(self.fixdim.values())
			fullx0[self.grididcs] = x0
		else:
			fullx0 = x0

		f, df = [np.asarray(val).flatten() for val in self.interpolate(self.resulttree, np.array(fullx0)[self.grididcs])]
		return {"f":f, "df":df} 

	def predictF(self, x0):
		return self.predict(x0)["f"]	

	def predictDF(self, x0):
		return self.predict(x0)["df"]	

def UniformCI(x0, tau, alpha, kernel, N, bw, Q, exogdens, endogdens, exogdenshat, endogdenshat, opol):
	# use f(x0) to calculate true quantile
	# combine the two to get a feasible sparsity
	# use information about kernel to construct variance and L(lambda) bound
	#print(list(zip(exogdens, x0)))
	#print(x0, Q(x0), endogdens(Q(x0)), np.product([d(v) for d,v in zip(exogdens, x0)]))
	useestimated = True
	if useestimated:
		fx = exogdenshat.pdf(x0) 
		sparsity = 1/np.array([dh.pdf(np.matrix(y), np.matrix(x0)).item() for dh,y in zip(endogdenshat, Q(x0))])
		sparsity /= 60 # to trim to true one which is .75
	else:
		fx = np.product([d(v) for d,v in zip(exogdens, x0)])
		sparsity = 1/endogdens(Q(x0)) 

	from etrics.Utilities import matrixbyelem
	X, Y = np.meshgrid(range(opol+1), range(opol+1))
	Np = matrixbyelem(X, Y, lambda x, y: kernel.moment(x+y))
	Tp = matrixbyelem(X, Y, lambda x, y: kernel.moment2(x+y))
	Qp = matrixbyelem(X, Y, lambda x, y: kernel.momentprime2(x+y))
	Npinv = np.linalg.inv(Np)
	C_0 = (Npinv * Qp * Npinv)[1,1]/(Npinv * Tp * Npinv)[1,1]
	L_0 = (-2 * np.log(bw)) * ((1 + (-2 * np.log(bw))) ** -1) * (-np.log(-np.log(1-alpha)/2) + np.log((C_0 ** (1/2)) / 2*np.pi) ) 
	var = tau * (1-tau) * (Npinv * Tp * Npinv)[0,0] * (sparsity ** 2) / (fx * N * bw) # from nh^-.5 from L_0  
	uniformbounds = np.vstack([Q(x0) - L_0*np.sqrt(var), Q(x0) + L_0*np.sqrt(var)]).T.tolist()
	#print(x0, sparsity, fx, L_0, np.sqrt(var), uniformbounds)
	return  uniformbounds

class RandomSystem:
	Specification = enum('Linear', 'Quadratic', 'Individual')
	
	@property
	def exog(self):
		return self._exog[self._ridcs,:]

	@property
	def endog(self):
		return self._endog[self._ridcs,:] 

	@property
	def names(self):
		return self._names
	
	@property
	def distributions(self):
		if len(self._dists) < len(self._names):
			raise Exception("don't know all distributions of the exogenous variables.")
		else:	
			return self._dists 
	
	@property
	def SampleExogenousDistribution(self):
		est = sm.nonparametric.KDEMultivariate(data=self.exog, var_type='c'*(self.exog.shape[1]), bw='normal_reference')
		return est

	@property
	def SampleConditionalMarginalDistributions(self):
		f = sm.nonparametric.KDEMultivariateConditional
		est = [f(self.endog[:,i].reshape((-1,1)), self.exog, 'c', 'c'*self.exog.shape[1], bw='normal_reference') for i in range(self.endog.shape[1])]
		return est	

	@property
	def errordistributions(self):
		if len(self._errordists) < len(self._unobsnames):
			raise Exception("don't know all distributions of the errors [{0}], # distributions: {1}".format(self._unobsnames, len(self._errordists)))
		else:	
			return self._errordists
	
	def __init__(self, N, K, dimY, theta0):
		self.theta0 = theta0
		self.freeDim = dimY
		self.dimY = dimY
		self.N = N
		self.K = K
		self._exog = np.empty((N, K))
		self._endog = np.empty((N, dimY))
		self._unobs = None 
		self._names = []
		self._dists = []
		self._errordists = []
		self._unobsnames = []
		self._ynames = []
		self.knownforms = {RandomSystem.Specification.Linear:self.Linear,\
			RandomSystem.Specification.Quadratic:self.Quadratic}
		self.ResetResamplingIndices()	
	
	def F0(self, spec):
		if spec in self.knownforms:
			return self.knownforms[spec]
		else:
			return self.Individual
	
	def SetResamplingIndices(self, idcs):
		self._ridcs = idcs

	def ResetResamplingIndices(self):
		self._ridcs = range(self.N)

	def ExogenousByName(self, name):
		return self.exog[:,self._names.index(name)].reshape(self.N, 1)

	def EndogenousByName(self, name):
		return self.endog[:,self._ynames.index(name)].reshape(self.N, 1)

	def GenerateObserved(self, dists, names):
		self._names += names
		self._dists += dists 
		for i in range(self.N):
			self._exog[i,:] = np.array([d.rvs() for d in dists])

	def GenerateUnobserved(self, dists, names):
		self._unobsnames = names 
		self._errordists += dists 
		if self._unobs is None:
			self._unobs = np.empty((self.N, len(dists)))

		for i in range(self.N):
			self._unobs[i,:] = np.array([d.rvs() for d in dists])
		
	def CalculateEndogenous(self, spec, names):
		self._ynames += names
		for i in range(self.N):
			self._endog[i,0:self.dimY] = (self.F0(spec))(self._exog[i,:], self._unobs[i,:], self.theta0)

	def AddConstrainedEndogenous(self, cons, name): 
		# row by row since it's more flexible and intuitive
		newcol = np.empty((self.N, 1))
		for i in range(self.N):
			newcol[i] = cons(self.exog[i], self.endog[i])
		self.AddEndogenous(newcol, name)
		
	def AddEndogenous(self, fixed, names):
		self._ynames += names
		self._endog = np.hstack([self._endog, fixed])
		self.dimY += fixed.shape[1] 

	def AddObserved(self, fixed, names, dists = []):
		self._names += names
		if dists == None:
			from etrics.Utilities import EstimatedDistWrapper
			est = sm.nonparametric.KDEUnivariate(fixed)
			est.fit()
			self._dists += [EstimatedDistWrapper(est)]		
		else:
			self._dists += dists 
		self._exog = np.hstack([self._exog, fixed])
		self.K += fixed.shape[1]
	
	# should be called after (or instead of) GenerateUnobserved
	def AddUnobserved(self, fixed, names, dists = []):
		self._unobsnames += names
		self._errordists += dists 
		self._unobs = fixed if self._unobs is None else np.hstack([self._unobs, fixed])

	def PrintDescriptive(self):
		print("Endogenous variables: ") 
		for f in [np.average, np.min, np.max]:
			print([self._ynames[i] +": "+str(f(self.endog[:,i])) for i in range(self.dimY)])
		print("Exogenous variables: ") 
		print([self._names[i] +": "+str(np.average(self.exog[:,i])) for i in range(self.K)])

	def Linear(self, xi, eps, theta):
		K = len(xi) # self.K
		return np.maximum( np.max(self._exog)*np.ones(K) \
			+ np.dot(theta, xi[0:K]) \
			+ eps[0:self.dimY], 0.)

	def Quadratic(self, xi, eps, theta):			
		K = len(xi) # self.K
		return np.dot(theta, xi[0:K]) \
			+ .2*np.dot(np.dot(xi[0:K].T, theta), xi[0:K]) \
			+ eps[0:self.dimY]
		#return (np.max(self._exog)**2)*np.ones(K) \
		#	+ np.dot(theta, xi[0:K]) \
		#	+ .2*np.dot(np.dot(xi[0:K].T, theta), xi[0:K]) \
		#	+ eps[0:self.dimY]
	
