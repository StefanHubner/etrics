#!/usr/bin/python3.3

from etrics.Utilities import EventHook

import scipy as sp
import numpy as np
import numexpr as ne

import statsmodels.api as sm
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import (resettable_cache, cache_readonly, cache_writable)
from statsmodels.tools.tools import rank

from matplotlib import rc
from numpy import arange, cos, pi
import pylab as plot

class NLRQ(base.LikelihoodModel):
	"""
		This implementation follows the algorithm proposed in 
		"An interior point algorithm for nonlinear quantile regression" 
	    by Roger Koenker, Beum J. Park in Journal of Econometrics (1996)
	"""		

	def __init__(self, endog, exog, **kwargs):
		super(NLRQ, self).__init__(endog, exog, **kwargs)
		self._initialize()
		self.PostEstimation = EventHook()
		self.PostInnerStep = EventHook()
		self.PostOuterStep = EventHook()

	def _initialize(self):
		self.nobs = float(self.endog.shape[0])
		self.df_resid = np.float(self.exog.shape[0] - self.parlen)
		self.df_model = np.float(self.parlen)
		
		self.eps = 10e-07
		self.maxit = 100
		self.beta = 0.97
		self.par = None
		self.linearinpar = False
		self.weights = np.ones(self.nobs)
		self.gradient = np.zeros(self.nobs * self.parlen).reshape(self.nobs, self.parlen)

	def setcontrol(self, maxit, eps):
		self.maxit = maxit
		self.eps = eps
	
	def fit(self, **kwargs):

		self.x0 = kwargs["x0"] # have to fit dimensions TODO check
		self.weights = kwargs["weights"].reshape(self.nobs) # have to fit dimensions TODO check
		self.par = np.zeros(self.parlen) # TODO smarter starting values

		w = np.zeros(self.nobs)
		self.wendog = np.multiply(self.endog, self.weights)

		snew = np.sum(self.loss(self.wendog - self.predict(self.par)))
		sold, lam, outer, inner, k = 10e+20, np.array(1.), 0, 0, 0

		while outer <= self.maxit and sold - snew > self.eps:
			if outer == 0 or not self.linearinpar:
				self.calculategradient(self.par)

			self.actstep, zw, k  = self.meketon(self.gradient, self.wendog - self.predict(self.par), w, tau = self.tau) 
			res = sp.optimize.minimize_scalar(self.step, bounds=(0., 1.), method='bounded')
			self.par += res.x * self.actstep
			sold, snew = snew, np.sum(self.loss(self.wendog - self.predict(self.par))) 

			q,r = sp.linalg.qr(self.gradient, mode='economic')
			w = zw - np.dot(self.gradient, np.dot(np.linalg.inv(r), np.dot(q.T, zw)))

			w1 = np.max(np.maximum(w, 0))
			if w1 > self.tau:
				w *= np.divide(self.tau, w1 + self.eps)
			w0 = np.max(np.maximum(-w, 0))
			if w0 > 1 - self.tau:
				w *= np.divide(1 - self.tau, w0 + self.eps)
			
			inner += k
			outer += 1
			self.PostOuterStep.Fire({"iteration":outer, "par":self.par, "sold":sold, "snew":snew, "stepsize":res.x})
				
		self.normalized_cov_params =  np.identity(self.parlen)
		res = NLRQResults(self, self.par, np.identity(self.parlen))
		res.fit_history['outer_iterations'] = outer
		res.fit_history['avg_inner_iterations'] = inner/outer 

		return NLRQResultsWrapper(res)

	def calculategradient(self, par):
		for i in range(int(self.nobs)):
			self.gradient[i,], self.linearinpar = self.Df(self.exog[i,], self.x0, self.par)
			self.gradient[i,] *= self.weights[i]

	def predict(self, params, exog=None):
		return np.dot(self.gradient, params)

	def loss(self, residuals):
		return self.tau * np.maximum(residuals, 0) - (1 - self.tau) * np.minimum(residuals, 0)

	def meketon(self, x, y, w, tau):
		yw = 10e+20
		k = 0
		z = None
		while k < self.maxit and yw - np.dot(y, w) > self.eps:
			d = np.minimum(tau - w, 1 - tau + w)

			wx, wy = np.multiply(x.T, d).T, np.multiply(y, d)
			q,r = sp.linalg.qr(wx, mode='economic')
			wbeta = np.dot(np.linalg.inv(r), np.dot(q.T, wy))
			wresid = y - np.dot(x, wbeta)
			
			yw = np.sum(self.loss(wresid))
			s = wresid * np.square(d) 
			alpha = np.max(np.concatenate([[self.eps], np.maximum( np.divide(s, tau - w), np.divide(-s, 1 - tau + w))]))
			w += self.beta/alpha * s
			k += 1
			self.PostInnerStep.Fire({"iteration":k, "par":wbeta, "yw":yw, "ydotw":np.dot(y,w)})

		return wbeta, w, k
	
	def step(self, lam):
		return np.sum(self.loss(self.wendog - self.predict(self.par + lam * self.actstep)))

class NLRQResults(base.LikelihoodModelResults):
	fit_history = {}

	def __init__(self, model, params, normalized_cov_params, scale = 1):
		super(NLRQResults, self).__init__(model, params, normalized_cov_params, scale)
		self.nobs = model.nobs
		self.df_model = model.df_model
		self.df_resid = model.df_resid

	@cache_readonly
	def fittedvalues(self):
		return self.model.predict(self.params)

	@cache_readonly
	def resid(self):
		return self.model.wendog - self.fittedvalues 

	@cache_readonly
	def varcov(self):
		return self.cov_params(scale=1.)

	@cache_readonly
	def pvalues(self):
		return sp.stats.norm.sf(np.abs(self.tvalues))*2

	def summary(self, yname=None, xname=None, title=0, alpha=0.05, return_fmt='text'):
		from statsmodels.iolib.summary import (summary_top, summary_params, summary_return) 
		top_left = [('Dep. Variable:', None), 
			('Model:', None), 
			('Method:', ['Interior Point']), 
			('Date:', None), 
			('Time:', None)] 

		top_right = [('No. Observations:', None), 
			('Df Residuals:', None), 
			('Df Model:', None),
			('Outer Iterations:', ["%d" % self.fit_history['outer_iterations']]), 
			('Avg. Inner Iterations:', ["%d" % self.fit_history['avg_inner_iterations']]) ]

		if not title is None:
			title = "Nonlinear Quantile Regression Results"
	
		from statsmodels.iolib.summary import Summary
		smry = Summary()
		smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
		smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=False)
		
		return smry


class NLRQResultsWrapper(lm.RegressionResultsWrapper):
	pass

wrap.populate_wrapper(NLRQResultsWrapper, NLRQResults)	

def LocalPolynomial2(xi, x0, par):
	K = int(1/2+np.sqrt(l-3/4))
	mu0 = par[0]
	mu1 = par[1:K+1]
	mu2 = par[K+1:].reshape(K, K)

	return mu0 + np.dot(mu1, xi - x0) + (xi - x0).T * mu2 * (xi - x0) 

def DLocalPolynomial2(xi, x0, par):
	return np.concatenate([[1], xi - x0, np.kron(xi - x0, xi - x0)]), True
	
def TraceOuter(info):
	print("{0}. outer iteration: sold = {1:.3f} snew = {2:.3f} par = {3} stepsize={4:.3f}"\
		.format(info["iteration"], info["sold"], info["snew"], info["par"], info["stepsize"])) 

def TraceInner(info):
	print("\t{0}. inner iteration: yw = {1:.3f} y.w = {2:.3f} dir = {3}"\
		.format(info["iteration"], info["yw"], info["ydotw"], info["par"]))

def grid(y, x, tau, h, size):
	parlen = x.shape[1] * (x.shape[1] + 1) + 1
	nlrqmodel = NLRQ(y, x, tau=tau, f=LocalPolynomial2, Df=DLocalPolynomial2, parlen=parlen)
	#nlrqmodel.PostOuterStep += TraceOuter;
	#nlrqmodel.PostInnerStep += TraceInner;

	for gp in np.linspace(np.min(x), np.max(x), num=size):
		weights = sp.stats.distributions.norm.pdf((x-gp)/h)
		nlrqresults = nlrqmodel.fit(x0 = gp, weights = weights) 
		yield np.concatenate([[gp], nlrqresults.params]).tolist()[0:(2+x.shape[1])]

	#xname = ("mu"+" mu".join(map(str, range(parlen)))).split()
	#yname = ["y"]
	#print(nlrqresults.summary(xname=xname, yname=yname))
			
def main():

	result = {} 
	dosimulation = True 
	dosomethingaboutit = False
	gridpoints = 35 
	bandwidth = 1 
	taus = [.1, .5, .9]

	if dosimulation:
		N = 600
		class data:
			exog = sp.stats.distributions.uniform.rvs(0, 4*sp.pi, N)
			endog = sp.sin(exog) + sp.stats.distributions.norm.rvs(0, 0.4, N) * exog/3
			#endog = sp.sin(exog) + sp.stats.distributions.chi2.rvs(3, -3, size=N) * exog/3
			exog = exog.reshape(N,1)
	else:
		data = sm.datasets.strikes.load()

	if dosomethingaboutit:
		for x, f0, Df0 in grid(data.endog, data.exog, 0.5, bandwidth, gridpoints):
			print(("{:+.4f} "*3).format(x, f0, Df0))
	else:
		for tau in taus:
			result[tau] = np.array(list(grid(data.endog, data.exog, tau, bandwidth, gridpoints)))

	fig=plot.figure(1, figsize=(9,13))
	plot.subplot(211)
	plot.plot(data.exog, data.endog, 'o')
	plot.grid(True)	
	for tau in taus:
		plot.plot(result[tau][:,0], result[tau][:,1], '-')
	plot.subplot(212)	
	plot.grid(True)	
	for tau in taus:
		plot.plot(result[tau][:,0], result[tau][:,2], '-')

	#fig.savefig('sin.pdf', dpi=fig.dpi, orientation='portrait', bbox_inches='tight', papertype='a4')
	#plot.show()


if __name__ == '__main__':
	main()
