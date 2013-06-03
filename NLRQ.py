from etrics.Utilities import EventHook
from scipy.optimize import minimize
from scipy import stats
from scipy import linalg

from statsmodels.tools.decorators import (resettable_cache, cache_readonly, cache_writable)
from statsmodels.tools.tools import rank

import numpy as np
import statsmodels.api as sm
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm

class NLRQ(base.LikelihoodModel):

	def __init__(self, endog, exog, **kwargs):
		super(NLRQ, self).__init__(endog, exog, **kwargs)
		self._initialize()
		self.PostEstimation = EventHook()

	def _initialize(self):
		self.nobs = float(self.endog.shape[0])
		self.df_resid = np.float(self.exog.shape[0] - rank(self.exog))
		self.df_model = np.float(rank(self.exog)-1)
		
		self.eps = 10e-07
		self.maxit = 10
		self.beta = 0.97
		self.par = None
		self.linearinpar = False
		self.gradient = zeros(self.nobs, self.parlen)

	def setcontrol(self, maxit, eps):
		self.maxit = maxit
		self.eps = eps
	
	def fit(self, **kwargs):

		self.x0 = kwargs["x0"] # have to fit dimensions TODO check
		self.par = zeros(self.parlen) # TODO smarter starting values

		w = zeros(self.nobs)
		snew = loss(self.endog - self.predict(self.par))
		sold = 10e+20
		lam = 1

		while k <= self.maxit and sold - snew > self.eps:
			if not self.linearinpar or k == 0:
				self.calculategradient(self.par)

			step, zw  = self.meketon(self.gradient, self.endog - self.predict(self.par), w, tau = self.tau) 
			lam # from optimization of step(), self.par updated inside
			self.par += lam*step
			sold, snew = snew, np.sum(loss(self.endog - self.predict(self.par))) 

			q,r = linalg.qr(self.gradient, mode='economic')
			w = zw - np.dot(np.linalg.inv(r), np.dot(q.T, zw))

			w1 = np.max(np.maximum(w, 0))
			if w1 > self.tau:
				w *= np.divide(self.tau, w1 + self.eps)
			w0 = np.max(np.maximum(-w, 0))
			if w0 > 1 - self.tau:
				w *= np.divide(1 - self.tau, w0 + self.eps)
			
			k += 1
				
		self.normalized_cov_params =  np.identity(self.parlen)
		res = NLRQResults(self, np.ones(self.exog.shape[1]), normalized_cov_params=self.normalized_cov_params)
		res.fit_history['outer_iterations'] = 10
		res.fit_history['avg_inner_iterations'] = 5.2

		return NLRQResultsWrapper(res)

	def calculategradient(self, par):
		for i in range(self.nobs):
			self.gradient[i,], self.linearinpar = Df(self.exog[i,], self.x0, self.par)

	def predict(self, params, exog=None):
		return dot(self.gradient, params)

	def loss(self, residuals):
			return tau * np.maximum(residuals, 0) - (1 - tau) * np.minimum(residuals, 0)

	def meketon(self, x, y, w, tau):
		yw = 10e+20
		k = 1
		z = None
		while k <= self.maxit and yw - np.dot(y, w) > self.eps:
			d = np.maximum(tau - w, 1 - tau + w])
			z = sm.WLS(y, x, weights= d**2).fit()
			yw = np.sum(self.loss(z.resid))
			k += 1
			s = z.resid * d**2
			alpha = max(self.eps, np.maximum( np.divide(s, tau - w), np.divide(-s, 1 - tau + w)))
			w += self.beta/alpha * s
		
		return z.coef, w
	
	def step(self, lambda, step, pars):
		self.par = pars + lambda * step
		return np.sum(self.loss(self.endog - self.predict(self.par)))

class NLRQResults(base.LikelihoodModelResults):
	fit_history = {}

	def __init__(self, model, params, normalized_cov_params, scale = 1):
		super(NLRQResults, self).__init__(model, params, normalized_cov_params, scale)
		self.nobs = model.nobs
		self.df_model = model.df_model
		self.df_resid = model.df_resid

	@cache_readonly
	def fittedvalues(self):
		return np.dot(self.model.exog, self.params)

	@cache_readonly
	def resid(self):
		return self.model.endog - self.fittedvalues 

	@cache_readonly
	def varcov(self):
		return self.cov_params(scale=1.)

	@cache_readonly
	def pvalues(self):
		return stats.norm.sf(np.abs(self.tvalues))*2

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
			('Inner Iterations:', ["%d" % self.fit_history['avg_inner_iterations']]) ]

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
	
def main():
	data = sm.datasets.longley.load()
	data.exog = sm.add_constant(data.exog, prepend=False)
	xname = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
	yname = ["y"]
	parlen = data.exog.shape[0] * (data.exog.shape[0] + 1) + 1
	qrresults = NLRQ(data.endog, data.exog, tau=0.9, f=LocalPolynomial2, Df=DLocalPolynomial2, parlen=parlen).fit() 
	print(qrresults.summary(xname=xname, yname=yname))

if __name__ == '__main__':
	main()
