from etrics.NLRQ import NLRQ, LocalPolynomial2, DLocalPolynomial2, TraceInner, TraceOuter
from etrics.Utilities import cartesian, Timing, enum
from scipy.stats import norm, scoreatpercentile
import numpy as np
import scipy as sp
from fractions import Fraction
import statsmodels.api as models
import collections

class NLRQSystem:
	def __init__(self, endog, exog, tau, componentwise):
		self._endog = endog
		self._exog = exog
		self._tau = tau
		self._globalresid = None
		self.parlen = self.exog.shape[1] * (self.exog.shape[1] + 1) + 1
		self.sizeperdim = 5 
		self.eps = 0.1
		self.componentwise = componentwise
		self.fitted = False
	
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
	def globalresiduals(self):
		if not self.fitted:
			print("fitgrid first")
		return self._globalresid
		
	def predict(self, x0):
		#print(x0)
		#print(self.results)
		grididcs = [i for i in range(self.exog.shape[1]) if i not in self.fixdim]
		#print(grididcs)
		if not self.fitted:
			print("fitgrid first")
		elif False in [x0[dim] == val for dim,val in self.fixdim.items()]:
			print("cannot estimate, one dimension was fixed at another value")
		else:
			tree = {}
			self.populatetree(tree, self.results, self.endog.shape[1], len(grididcs)*self.endog.shape[1])
			return self.interpolate(tree, np.array(x0)[grididcs])[0]
			#return np.zeros(self.endog.shape[1])
	
	def populatetree(self, t, z, sizey, sizedy):
		if z.shape[1] > sizey+sizedy:
			for v in np.unique(z[:,0]):
				t[v] = {}
				self.populatetree(t[v], z[z[:,0]==v][:,1:], sizey, sizedy)
		else:
			t['f'] = z[:,:sizey]
			t['df'] = z[:,-sizedy:]

	def interpolate(self, node, x0):
		if len(x0) > 0:
			snode = np.array(sorted(node))
			idx = snode.searchsorted(x0)
			lx, ux = snode[idx-1][0], snode[idx][0]
			ly = self.interpolate(node[lx], x0[1:])
			uy = self.interpolate(node[ux], x0[1:])
			return ly+(x0[0]-lx)*(uy-ly)
		else:
			return node['f']
	
	def fit(self, x0, weights):
		M = self.endog.shape[1]
		#Omega = np.matrix(cartesian([np.linspace(0+self.eps,1-self.eps,self.sizeperdim).tolist()]*M))
		Omega = self.gridball(M, self.sizeperdim) if not self.componentwise else np.identity(M)
		L = Omega.shape[0]
		K = self.exog.shape[1]
		Lambda = np.zeros(L*(K+1)).reshape(L, K+1)
		self.resid = np.zeros(self.exog.shape[0]*L).reshape(self.exog.shape[0], L)

		for i in range(L):
			y = np.dot(self.endog, Omega[i,:].T).reshape(self.endog.shape[0])
			nlrqmodel = NLRQ(y, self.exog, tau=self.tau, f=LocalPolynomial2, Df=DLocalPolynomial2, parlen=self.parlen)

			#nlrqmodel.PostOuterStep += TraceOuter;
			#nlrqmodel.PostInnerStep += TraceInner;
			nlrqresults = nlrqmodel.fit(x0 = x0, weights = weights) 
			Lambda[i,:] = nlrqresults.params[0:K+1]
			self.resid[:,i] = nlrqresults.resid.T
	
		b = Lambda.T.reshape(L*(K+1)) # b=vec(Lambda)
		A = np.kron(np.identity(K+1), Omega) 
		q,r = sp.linalg.qr(A, mode='economic')
		x = np.dot(np.linalg.inv(r), np.dot(q.T, b)) # vec(mu) with mu = [mu0, mu1]
		mu = x.reshape(K+1, M).T
		mu0, mu1 = mu[:,0], np.delete(mu, 0, 1)
		#print(mu0)
		#print(mu1)

		return mu0, mu1 

	def fitgrid(self, h, sizeperdim, fixdim = {}): #TODO check dim vs dimy
		#dim = self.exog.shape[1] - len(fixdim)
		x0 = np.empty(self.exog.shape[1])
		grididcs = [i for i in range(self.exog.shape[1]) if i not in fixdim]
		dimy = self.endog.shape[1]
		#xmins, xmaxs = np.min(self.exog, axis=0), np.max(self.exog, axis=0)
		xq10s, xq90s = np.percentile(self.exog, 10, axis=0), np.percentile(self.exog, 90, axis=0)
		grid = []
		j = 0
		for i in grididcs:
			grid.append(np.linspace(xq10s[i], xq90s[i], sizeperdim))
		glen = len(cartesian(grid))
		residweights = np.empty((self.exog.shape[0], glen)) 
		resids = np.empty((self.exog.shape[0], dimy*glen)) 
		cgrid = cartesian(grid)
		self.results = np.empty((cgrid.shape[0], len(grididcs)+dimy+dimy*len(grididcs)))
		for x0r in cgrid:
			x0[grididcs] = x0r
			if len(fixdim) > 0: x0[list(fixdim.keys())] = list(fixdim.values())
			dist = np.sum(np.abs(self.exog-x0)**2, axis=1)**.5
			weights = sp.stats.distributions.norm.pdf(dist/h)
			residweights[:,j] = weights
			with Timing("nlrqsystem({0}) at {1}: grid 10^{2}".format(self.tau, x0, self.exog.shape[1]), True):
				fct, grad = self.fit(x0 = x0, weights = weights) 
				#print(grad)
				#resids[:,j*dimy:(j+1)*dimy] = self.resid
				resids[:,np.array(range(dimy))*glen+j] = self.resid
				#yield np.concatenate([x0r, fct, np.delete(grad, list(fixdim.keys()), 1).flatten(0)]).tolist()
				self.results[j,:] = np.concatenate([x0r, fct, np.delete(grad, list(fixdim.keys()), 1).flatten(0)])
			j += 1	
		#print(resids.shape)	
		residweights = np.divide(residweights.T, np.sum(residweights, axis=1)).T
		self._globalresid = np.dot(np.multiply(resids, np.kron(np.ones((1,dimy)), residweights)), \
			np.kron(np.identity(dimy), np.ones((glen, 1))))
		#self._globalresid = np.dot(resids, np.kron(np.identity(dim), residweights.T))
		#print(self._globalresid.shape)
		#print (np.sum(residweights, axis=0))
		self.fitted = True
		self.fixdim = fixdim
	
	def GetResults(self):
		if not self.fitted:
			print("fitgrid first")
		return self.results

	def gridball(self, dimensions, sizeperdim):
		eps = 0.05
		x2y2=cartesian([np.linspace(0+eps,1-eps,sizeperdim)**2]*(dimensions-1)).tolist()
		for i in x2y2: i.append(1-np.sum(i))
		x2y2=np.array(x2y2)
		return np.sqrt(x2y2[np.where(np.all(x2y2>=0, axis=1))])
				
class RandomSystem:
	Specification = enum('Linear', 'Quadratic', 'Individual')
	
	@property
	def exog(self):
		return self._exog

	@property
	def endog(self):
		return self._endog
	
	def __init__(self, N, K, dimY, theta0):
		self.theta0 = theta0
		self.dimY = dimY
		self.N = N
		self.K = K
		self._exog = np.empty((N, K))
		self._endog = np.empty((N, dimY))
		self._unobs = np.empty((N, dimY))
		self._names = []
		self._ynames = []
		self.knownforms = {RandomSystem.Specification.Linear:self.Linear,\
			RandomSystem.Specification.Quadratic:self.Quadratic}
	
	def F0(self, spec):
		if spec in self.knownforms:
			return self.knownforms[spec]
		else:
			return self.Individual
	
	def ExogenousByName(self, name):
		return self.exog[:,self._names.index(name)].reshape(self.N, 1)

	def EndogenousByName(self, name):
		return self.endog[:,self._ynames.index(name)].reshape(self.N, 1)

	def GenerateObserved(self, dists, names):
		self._names += names
		for i in range(self.N):
			self._exog[i,:] = np.array([pdf() for pdf in dists])

	def GenerateUnobserved(self, dists):
		for i in range(self.N):
			self._unobs[i,:] = np.array([pdf() for pdf in dists])
		
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
		self.dimY += 1

	def AddObserved(self, fixed, names):
		self._names += names
		self._exog = np.hstack([self._exog, fixed])
		self.K += fixed.shape[1]
	
	def AddUnobserved(self, fixed):
		self._unobs = np.hstack([self._unobs, fixed])

	def PrintDescriptive(self):
		print("Endogenous variables: ") 
		print([self._ynames[i] +": "+str(np.average(self.endog[:,i])) for i in range(self.dimY)])
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
	
