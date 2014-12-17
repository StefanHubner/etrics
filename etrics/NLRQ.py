from etrics.Utilities import EventHook, Timing

import scipy as sp
import numpy as np
import numexpr as ne

import statsmodels.api as sm
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm
from statsmodels.tools.decorators import (resettable_cache, cache_readonly, cache_writable)
from statsmodels.tools.tools import rank

from matplotlib import rc, cm
from numpy import arange, cos, pi
import logging

class NLRQ(base.LikelihoodModel):
	"""
		This implementation follows the algorithm proposed in 
		"An interior point algorithm for nonlinear quantile regression" 
	    by Roger Koenker, Beum J. Park in Journal of Econometrics (1996)
	"""		

	def __init__(self, endog, exog, **kwargs):
		super(NLRQ, self).__init__(endog, exog, **kwargs)
		ne.set_num_threads(8)
		self._initialize()
		self.PreEstimation = EventHook()
		self.PostVarianceCalculation = EventHook()
		self.PostEstimation = EventHook()
		self.PostInnerStep = EventHook()
		self.PostOuterStep = EventHook()

	def _initialize(self):
		self.nobs = float(self.endog.shape[0])
		self.df_resid = np.float(self.exog.shape[0] - self.parlen)
		self.df_model = np.float(self.parlen)
		
		self.eps = 10e-06
		self.epsinner = 10e-03
		self.maxit = 100
		self.beta = 0.97
		self.par = None
		self.linearinpar = False
		self.UseQR = False
		self.normalizeddensityestimation = False 
		self.weights = np.ones(self.nobs)
		self.gradient = np.zeros(self.nobs * self.parlen).reshape(self.nobs, self.parlen)
		if self.normalizeddensityestimation:
			self.estimatedensitiesnormalized()
		else:	
			self.estimatedensities()

	def setcontrol(self, maxit, eps):
		self.maxit = maxit
		self.eps = eps
	
	def estimatedensitiesnormalized(self):		
		# normalization: must also be normalized at calling pdf (varcov calculation), not x0 for conditional
		self.exogL, self.endogL = np.linalg.cholesky(np.matrix(np.cov(self.exog.T))), np.linalg.cholesky(np.matrix(np.cov(self.endog.T)))
		exogZ = np.dot(np.matrix(self.exog) - np.mean(self.exog, axis=0), np.linalg.inv(self.exogL))
		endogZ = np.dot(np.matrix(self.endog).T - np.mean(self.endog, axis=0), np.linalg.inv(self.endogL))
		self._exogdens = sm.nonparametric.KDEMultivariate(data=exogZ, var_type=self.exogtype, bw='normal_reference')
		self._endogdens = sm.nonparametric.KDEMultivariateConditional(endogZ, self.exog, self.endogtype, self.exogtype, bw='normal_reference')
	
	def estimatedensities(self):		
		self._exogdens = sm.nonparametric.KDEMultivariate(data=self.exog, var_type=self.exogtype, bw='normal_reference')
		self._endogdens = sm.nonparametric.KDEMultivariateConditional(self.endog, self.exog, self.endogtype, self.exogtype, bw='normal_reference')
	
	def fit(self, **kwargs):
		self.x0 = kwargs["x0"] 
		self.dist = kwargs["dist"] 
		self.kernel = kwargs["kernel"] 
		self.bw = kwargs["bw"] 
		H_k = (1/self.bw)**self.exog.shape[1]
		self.weights = H_k * self.kernel.pdfnorm(self.dist/self.bw).reshape(self.nobs) # TODO: pdf vs pdfnorm
		self.PreEstimation.Fire({"nonzero": 100*np.mean(np.abs(self.weights) > 0)})
		self.par = np.zeros(self.parlen) # TODO smarter starting values

		w = np.zeros(self.nobs)
		self.wendog = ne.evaluate("y * w", local_dict = {'y': self.endog, 'w': self.weights})
		unew = ne.evaluate("y - yhat", local_dict = {'y': self.wendog, 'yhat':self.predictlinear(self.par)})
		snew = np.sum(self.loss(unew))

		sold, lam, outer, inner, k = 10e+40, np.array(1.), 0, 0, 0

		while outer <= self.maxit and sold - snew > self.eps:
			if outer == 0 or not self.linearinpar:
				self.calculategradient(self.par)

			self.actstep, zw, k  = self.meketon(self.gradient, unew, w, tau = self.tau) 

			res = sp.optimize.minimize_scalar(self.step, bounds=(0., 1.), method='bounded')
			self.par += res.x * self.actstep
			unew = ne.evaluate("y - yhat", local_dict = {'y': self.wendog, 'yhat':self.predictlinear(self.par)})
			sold, snew = snew, np.sum(self.loss(unew))

			if self.UseQR:
				q,r = sp.linalg.qr(self.gradient, mode='economic')
				wbeta = np.dot(np.linalg.inv(r), np.dot(q.T, zw))
			else:
				wbeta = sp.linalg.lstsq(self.gradient, zw)[0]
			
			w = ne.evaluate("zw - zwhat", local_dict = {'zw':zw, 'zwhat':np.dot(self.gradient, wbeta)})

			w1 = np.max(w) # original: w1 = np.max(np.maximum(w, 0))
			if w1 > self.tau:
				w = ne.evaluate("tau * w / w1", local_dict = {'w': w, 'tau':self.tau, 'w1':w1 + self.eps})
			w0 = -np.min(w) # original: w0 = np.max(np.maximum(-w, 0))
			if w0 > 1 - self.tau:
				w = ne.evaluate("(1-tau) * w / w0", local_dict = {'w': w, 'tau':self.tau, 'w0':w0 + self.eps})
				
			inner += k
			outer += 1
			self.PostOuterStep.Fire({"iteration":outer, "par":self.par, "sold":sold, "snew":snew, "stepsize":res.x, "inner":k})
				
		self.normalized_cov_params =  self.varcov
		res = NLRQResults(self, self.par, self.normalized_cov_params)
		res.fit_history['outer_iterations'] = outer
		res.fit_history['avg_inner_iterations'] = inner/outer 

		return NLRQResultsWrapper(res)
	
	@property
	def varcov(self): 
		H_N = self.exog.shape[0] * (self.bw) ** self.exog.shape[1]
		# normalization: must also be normalized at density estimation, not x0 for conditional
		if self.normalizeddensityestimation:
			exogZ = np.dot(self.x0 - np.mean(self.exog, axis=0), np.linalg.inv(self.exogL))
			endogZ = np.dot(self.par[0] - np.mean(self.endog, axis=0), np.linalg.inv(self.endogL))
			fx = self._exogdens.pdf(exogZ) / np.linalg.det(self.exogL)
			fyx = self._endogdens.pdf(np.matrix(endogZ), np.matrix(self.x0)) / np.linalg.det(self.endogL)
		else:
			fx = self._exogdens.pdf(self.x0) 
			fyx = self._endogdens.pdf(np.matrix(self.par[0]), np.matrix(self.x0))
			
		vcov = self.tau * (1-self.tau) * self.kernel.B1 / (fyx**2 * fx * H_N)
		Hsq = np.hstack([[1], self.kernel.sigmas**2 * self.bw ** 2]) # z = [1, pi/h] and g = [g0,h*g1] in bahadur
		# hence divide variance by h^2 (including variance term (not caputred by kernel here))
		vcov /= Hsq
		self.PostVarianceCalculation.Fire({"H_N":H_N, "B1":np.diag(self.kernel.B1), "sparsity":(1/(fyx**2)) , "density": fx, "stderrs": np.sqrt(np.diagonal(vcov))})
		return vcov

	def calculategradient(self, par):
		self.gradient, self.linearinpar = self.Df(self.exog, self.x0, self.par)
		self.gradient *= self.weights.reshape(self.weights.shape[0], 1)

	def predictlinear(self, params):
		return np.dot(self.gradient, params)
	
	def residuals(self, params):
		#return self.endog - np.dot(self.gradient, params)
		return self.wendog - self.predictlinear(params)
	
	def predict(self, params, exog = None):
		if exog is None:
			exog = self.exog
		return self.f(exog, self.x0, params) 

	def loss(self, residuals):
		return ne.evaluate("u * (tau - (u < 0))", local_dict = {'tau':self.tau, 'u':residuals})
		#return self.tau * np.maximum(residuals, 0) - (1 - self.tau) * np.minimum(residuals, 0)

	def meketon(self, x, y, w, tau):
		yw = np.Infinity 
		k = 0
		z = None
		while k < self.maxit and yw - ne.evaluate("sum(y*w)", local_dict = {'y':y, 'w':w }) > self.epsinner:
			# d = np.minimum(tau - w, 1 - tau + w)
			d = ne.evaluate('lhs + delta*(delta < 0)', local_dict = {'lhs':ne.evaluate("tau - w"), 'delta':ne.evaluate("1-2*(tau-w)")})

			# wx, wy = np.multiply(x.T, d).T, np.multiply(y, d)
			wx, wy = ne.evaluate("x * d", local_dict = {'x':x.T, 'd':d}).T, ne.evaluate("y * d", local_dict={'y':y, 'd':d})
			if self.UseQR:
				q,r = sp.linalg.qr(wx, mode='economic')
				wbeta = np.dot(np.linalg.inv(r), np.dot(q.T, wy))
			else:
				wbeta = sp.linalg.lstsq(wx, wy)[0]
			
			wresid = ne.evaluate("y - yhat", local_dict = {'y':y, 'yhat':np.dot(x, wbeta)})
			
			# yw = np.sum(self.loss(wresid))
			yw = ne.evaluate("sum(r)", local_dict={'r':self.loss(wresid)})
			s = ne.evaluate("wresid * d**2") 
			# alpha = np.max(np.concatenate([[self.eps], np.maximum( np.divide(s, tau - w), np.divide(-s, 1 - tau + w))]))
			alpha = max(self.eps, np.max(ne.evaluate("lhs + (rhs - lhs)*(rhs - lhs > 0)", local_dict= {'lhs':ne.evaluate("s/(tau-w)"), 'rhs':ne.evaluate("-s/(1-tau+w)")})))
			w += self.beta/alpha * s
			k += 1
			self.PostInnerStep.Fire({"iteration":k, "par":wbeta, "yw":yw, "ydotw":np.dot(y,w)})

		return wbeta, w, k
		
	def step(self, lam):
		return np.sum(self.loss(self.wendog - self.predictlinear(self.par + lam * self.actstep)))

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
		#return self.model.wendog - self.fittedvalues 
		return self.model.residuals(self.params) 

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

def Polynomial1(x, x0, par):
	K = x.shape[1]-1
	mu0 = par[0]
	mu1 = par[1:K+1].reshape(K, 1)
	return (mu0 + np.dot(x-x0, mu1)).reshape(x.shape[0])

def DPolynomial1(x, x0, par):
	return np.concatenate([np.ones((x.shape[0], 1)), x-x0], axis=1), True	

def Polynomial2(x, x0, par):
	K = int(1/2+np.sqrt(x.shape[1]-3/4))
	mu0 = par[0]
	mu1 = par[1:K+1].reshape(K, 1)
	mu2 = par[K+1:].reshape(K, K)
	return (mu0 + np.dot(x-x0, mu1) + np.sum(np.multiply(np.dot(x-x0, mu2), x-x0), axis=1).reshape(x.shape)).reshape(x.shape[0])

def DPolynomial2(x, x0, par):
	XkronX = np.multiply(np.kron(x-x0, np.ones(x.shape[1]).reshape(1,x.shape[1])), \
		np.kron(np.ones(x.shape[1]).reshape(1,x.shape[1]), x-x0))
	return np.concatenate([np.ones(x.shape[0]).reshape(x.shape[0], 1),x-x0,XkronX], axis=1), True	

def TracePreEstimation(logger, info):
	logger.debug("Proportion of nonzero weights: {0}%".format(info["nonzero"]))

def TraceVarianceCalculation(logger, info):
	logger.debug("VarCov: H_N = {} B1 = {} f_yx^-2 = {} f_x = {} stderrs = {}".format(info["H_N"], info["B1"], info["sparsity"], info["density"], info["stderrs"]))

def TraceOuter(logger, info):
	logger.debug("{0}. outer iteration: sold = {1:.3f} snew = {2:.3f} par = {3} stepsize={4:.3f} innersteps={5}"\
		.format(info["iteration"], info["sold"], info["snew"], info["par"], info["stepsize"], info["inner"])) 

def TraceInner(logger, info):
	logger.debug("\t{0}. inner iteration: yw = {1:.3f} y.w = {2:.3f} dir = {3}"\
		.format(info["iteration"], info["yw"], info["ydotw"], info["par"]))

def grid1d(y, x, tau, h, size):
	parlen1 = x.shape[1] + 1
	parlen2 = x.shape[1] * (x.shape[1] + 1) + 1
	from etrics.Utilities import TriangularKernel
	from etrics.NLRQSystem import UCIConstant
	k = TriangularKernel(scale=5.)
	nlrqmodel = NLRQ(y, x, tau=tau, f=Polynomial1, Df=DPolynomial1, parlen=parlen1, kernel=k)
	#nlrqmodel.PostOuterStep += TraceOuter;
	#nlrqmodel.PostInnerStep += TraceInner;
	fx = sm.nonparametric.KDEUnivariate(x)
	fx.fit()
	T0 = k.Tp[0,0]
	fyx = sm.nonparametric.KDEMultivariateConditional(endog=y, exog=x, dep_type='c', \
		indep_type='c', bw='normal_reference') 

	for gp in np.linspace(np.min(x), np.max(x), num=size):
		dist = np.sum(np.abs(x-gp)**2,axis=1)**.5
		#weights = sp.stats.distributions.norm.pdf(dist/h)/h # TODO: /h
		#with Timing("fit"):
		nlrqresults = nlrqmodel.fit(x0 = gp, kernel = sp.stats.distributions.norm, dist = dist, bw = h) 
		fx0 = fx.evaluate(gp).item()
		fyx0 = fyx.pdf(np.matrix(fx0), np.matrix(gp)).item()
		stdev = UCIConstant(0.05, h, T0) * np.sqrt(tau * (1-tau) * T0 / ( fyx0**2 * fx0 * y.shape[0] * h))
		yield np.concatenate([[gp], nlrqresults.params, [stdev]]).tolist() # [0:(2+x.shape[1])]

	#xname = ("mu"+" mu".join(map(str, range(parlen)))).split()
	#yname = ["y"]

def main():
	import pylab as plot

	result = {} 
	dosimulation = True 
	dosomethingaboutit = False
	gridpoints = 25 
	bandwidth = .8
	taus = [.1, .5, .9]
	c = dict(zip(taus, ['b', 'r', 'g']))

	if dosimulation:
		N = 600
		class data:
			exog = sp.stats.distributions.uniform.rvs(0, 4*sp.pi, N)
			endog = sp.sin(exog) + sp.stats.distributions.norm.rvs(0, 0.4, N) * (exog**0.5)
			exog = exog.reshape(N, 1)
	else:
		data = sm.datasets.strikes.load()

	if dosomethingaboutit:
		for x, f0, Df0 in grid1d(data.endog, data.exog, 0.5, bandwidth, gridpoints):
			print(("{:+.4f} "*5).format(x, f0, Df0, fx0, fyx0))
	else:
		for tau in taus:
			result[tau] = np.array(list(grid1d(data.endog, data.exog, tau, bandwidth, gridpoints)))

	fig=plot.figure(1, figsize=(9,13))
	plot.subplot(211)
	plot.plot(data.exog, data.endog, 'o')
	plot.grid(True)	
	for tau in taus:
		plot.plot(result[tau][:,0], result[tau][:,1], '-', c=c[tau])
		plot.plot(result[tau][:,0], result[tau][:,1] + result[tau][:,3], '--', c=c[tau])
		plot.plot(result[tau][:,0], result[tau][:,1] - result[tau][:,3], '--', c=c[tau])
	plot.subplot(212)	
	plot.grid(True)	
	for tau in taus:
		plot.plot(result[tau][:,0], result[tau][:,2], '-', c=c[tau])

	fig.savefig('sin.pdf', dpi=fig.dpi, orientation='portrait', bbox_inches='tight', papertype='a4')
	plot.show()

if __name__ == '__main__':
	main()
