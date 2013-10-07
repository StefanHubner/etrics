#!/usr/bin/python3

from etrics.NLRQ import NLRQ
from etrics.Utilities import cartesian, Timing, enum
from scipy.stats import norm, scoreatpercentile
import numpy as np
import scipy as sp
from fractions import Fraction
import statsmodels.api as models


class NLRQSystem:
	def __init__(self, endog, exog, tau, parlen, componentwise):
		self.endog = endog
		self.exog = exog
		self.tau = tau
		self.parlen = parlen
		self.sizeperdim = 5 
		self.eps = 0.1
		self.componentwise = componentwise

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
			nlrqmodel = NLRQ(y, self.exog, tau=self.tau, f=self.LocalPolynomial2, Df=self.DLocalPolynomial2, parlen=self.parlen)
			nlrqresults = nlrqmodel.fit(x0 = x0, weights = weights) 
			Lambda[i,:] = nlrqresults.params[0:K+1]
			self.resid[:,i] = nlrqresults.resid.T
	
		b = Lambda.T.reshape(L*(K+1)) # b=vec(Lambda)
		A = np.kron(np.identity(K+1), Omega) 
		q,r = sp.linalg.qr(A, mode='economic')
		x = np.dot(np.linalg.inv(r), np.dot(q.T, b)) # vec(mu) with mu = [mu0, mu1]
		mu = x.reshape(K+1, M).T
		mu0, mu1 = mu[:,0], np.delete(mu, 0, 1)
		print(mu0)
		print(mu1)

		return x

	def residuals(self):
		return self.resid

	def gridball(self, dimensions, sizeperdim):
		eps = 0.05
		x2y2=cartesian([np.linspace(0+eps,1-eps,sizeperdim)**2]*(dimensions-1)).tolist()
		for i in x2y2: i.append(1-np.sum(i))
		x2y2=np.array(x2y2)
		return np.sqrt(x2y2[np.where(np.all(x2y2>=0, axis=1))])
				
	def LocalPolynomial2(self, x, x0, par):
		K = int(1/2+np.sqrt(x.shape[1]-3/4))
		mu0 = par[0]
		mu1 = par[1:K+1].reshape(K, 1)
		mu2 = par[K+1:].reshape(K, K)
		return (mu0 + np.dot(x-x0, mu1) + np.sum(np.multiply(np.dot(x-x0, mu2), x-x0), axis=1).reshape(x.shape)).reshape(x.shape[0])
	
	def DLocalPolynomial2(self, x, x0, par):
		XkronX = np.multiply(np.kron(x-x0, np.ones(x.shape[1]).reshape(1,x.shape[1])), \
			np.kron(np.ones(x.shape[1]).reshape(1,x.shape[1]), x-x0))
		return np.concatenate([np.ones(x.shape[0]).reshape(x.shape[0], 1),x-x0,XkronX], axis=1), True	

def grid(y, x, tau, h, sizeperdim):
	parlen = x.shape[1] * (x.shape[1] + 1) + 1
	nlrqsystemmodel = NLRQSystem(y, x, tau, parlen, True)
	dim = x.shape[1]
	xmins, xmaxs = np.min(x, axis=0), np.max(x, axis=0)
	grid = []
	for i in range(dim): 
		grid.append(np.linspace(xmins[i], xmaxs[i], sizeperdim))
	# print(len(cartesian(grid)))	
	
	for x0 in cartesian(grid):
		dist = np.sum(np.abs(x-x0)**2, axis=1)**.5
		weights = sp.stats.distributions.norm.pdf(dist/h)
		with Timing("nlrqsystem({0}) at {1}: grid 10^{2}".format(tau, x0, x.shape[1])):
			par = nlrqsystemmodel.fit(x0 = x0, weights = weights) 
			yield np.concatenate([x0, par]).tolist()

class RandomSystem:
	Specification = enum('Linear', 'Quadratic', 'Individual')
	
	@property
	def exog(self):
		return self._exog

	@property
	def endog(self):
		return self._endog

	def __init__(self, N, K, theta0):
		self.domain = [1,3]
		self.theta0 = theta0
		self.K = self.Ktot = K
		self.N = N
		self._exog = np.zeros(N*K).reshape(N, K)
		self._endog = np.zeros(N*K).reshape(N, K)
		self.knownforms = {RandomSystem.Specification.Linear:self.Linear, RandomSystem.Specification.Quadratic:self.Quadratic}
	
	def F0(self, spec):
		if spec in self.knownforms:
			return self.knownforms[spec]
		else:
			return self.Individual
	
	def RegisterNewFunctionalForm(self):
		self.knownforms[RandomSystem.Specification.Individual] = self.Individual
			
	def Generate(self, spec, vcov):
		for i in range(self.N):
			self._exog[i,0:self.K] = sp.stats.distributions.uniform.rvs(self.domain[0], self.domain[1]-self.domain[0], self.K)
			eps = np.random.multivariate_normal([0]*self.K, vcov) if vcov is not None else sp.stats.distributions.norm.rvs(0, 0.4, self.K)
			
			self._endog[i,:] = (self.F0(spec))(self._exog[i,:], self.theta0 + (eps if vcov is not None else 0)) + (eps if vcov is None else 0)

	def AddFixed(self, fixed):
		self._exog = np.hstack([self._exog, fixed])
		self.Ktot = self.exog.shape[1] - self.K

	def Linear(self, xi, theta):
		return np.maximum( self.domain[1]*np.ones(self.K) + np.dot(theta, xi[0:self.K]), 0.)

	def Quadratic(self, xi, theta):			
		return (self.domain[1]**2)*np.ones(self.K) + np.dot(theta, xi[0:self.K]) + .2*np.dot(np.dot(xi[0:self.K].T, theta), xi[0:self.K]) 
