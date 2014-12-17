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

class RandomSystem:
	Specification = enum('Linear', 'Quadratic', 'Individual')
	
	@property
	def exog(self):
		return self._exog[self._ridcs,:]

	@property
	def exog1(self):
		return (self._exog - np.mean(self._exog, axis=0))/np.std(self._exog, axis=0)

	@property
	def exogtype(self):
		return "c"*self.exog.shape[1] 

	@property
	def endog(self):
		return self._endog[self._ridcs,:] 

	@property
	def endog1(self):
		return (self._endog - np.mean(self._endog, axis=0))/np.std(self._endog, axis=0)

	@property
	def endogtype(self):
		return "c"*self.endog.shape[1] 

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
		return self._exogdens 

	@property
	def SampleConditionalMarginalDistributions(self):
		return self._endogens

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
		self._exogdens = None
		self._endogdens = None
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

	def PrintDescriptive(self, logger):
		logger.info("Endogenous variables: ") 
		for f in [np.average, np.min, lambda x: np.percentile(x, 10), lambda x: np.percentile(x, 90),  np.max]:
			logger.info([self._ynames[i] +": "+str(f(self.endog[:,i])) for i in range(self.dimY)])
		logger.info("Exogenous variables: ") 
		logger.info([self._names[i] +": "+str(np.average(self.exog[:,i])) for i in range(self.K)])

	def EstimateDistributions(self):
		self._exogdens = lambda x: sm.nonparametric.KDEMultivariate(data=self.exog1, \
			var_type='c'*self.exog.shape[1], bw='normal_reference').pdf((x-np.mean(self.exog, axis=0))/np.std(self.exog, axis=0))
		k = sm.nonparametric.KDEMultivariateConditional
		f = lambda i: k(self.endog1[:,i].reshape((-1,1)), self.exog, 'c', 'c'*self.exog.shape[1], bw='normal_reference')
		self._endogens = [lambda y,x: f(i).pdf(np.matrix((y - np.mean(self.endog[:,i]))/np.std(self.endog[:,i])), \
			np.matrix(x)) for i in range(self.endog.shape[1])]
	
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
	
