from etrics.Utilities import EventHook, Timing
from scipy.optimize import minimize
from scipy import stats

from statsmodels.tools.decorators import (resettable_cache, cache_readonly, cache_writable)
from statsmodels.tools.tools import rank

import numpy as np
import scipy as sp
import numexpr as ne
import statsmodels.api as sm
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm

class quantreg(base.LikelihoodModel):
	""" 
	Interior Point Method: Portnoy, Koenker 1997
	The Gaussian hare and the Laplacian Tortoise

	min_x {c'x | Ax = b, 0 <= x <= u }
	"""	

	def __init__(self, endog, exog, **kwargs):
		super(quantreg, self).__init__(endog, exog, **kwargs)
		self.PostEstimation = EventHook()
		self.PreEstimation = EventHook()
		self.PostVarianceCalculation = EventHook()
		self.nobs = float(self.endog.shape[0])
		self.df_resid = np.float(self.exog.shape[0] - rank(self.exog))
		self.df_model = np.float(rank(self.exog)-1)

	def _initialize(self):
		informative = (self.weights != 0) if self.isnonparametric else np.repeat(True, self.nobs)
		self.c = -self.endog1[informative]
		self.A = self.exog1[informative,:].T
		self.b = (1-self.tau) * sum(self.exog1, 0)
		self.u = np.ones(self.c.shape[0])
		self.p  = (1-self.tau) * self.u # start on boundary
		self.beta = 0.9995 
		self.eps = 10e-05
		self.maxit = 50 
		self.big = 10e+20
	
	def fit(self, **kwargs): # kwargs x0 = x0, kernel = kernel, dist = dist, bw = bw
		self.isnonparametric = all([k in kwargs.keys() for k in ["x0", "kernel", "dist", "bw"]])
		if self.isnonparametric:
			self.kernel = kwargs.get("kernel")
			self.bw = kwargs.get("bw") 
			self.x0 = kwargs.get("x0")
			H_k = (1/self.bw)**self.exog.shape[1]
			self.weights = self.kernel.pdf(kwargs.get("dist")/self.bw).reshape(self.nobs) # TODO: pdf vs pdfnorm, *H_k ?
			self.endog1 = ne.evaluate("y * w", local_dict = {'y': self.endog, 'w': self.weights})
			const = np.isclose(np.max(self.exog, axis=0) - np.min(self.exog, axis=0))
			if const.any():
				self.exog = self.exog[:,~const]
			self.H = np.linalg.cholesky(np.matrix(np.cov(self.exog.T))) * self.bw
			self.exog1 = np.concatenate([np.ones((self.exog.shape[0], 1)), (self.exog - self.x0)/self.bw], axis=1) * self.weights.reshape((self.nobs,1))
			self.PreEstimation.Fire({"nonzero": 100*np.mean(np.abs(self.weights) > 0)})
			#print(100*np.mean(np.abs(self.weights) > 0))
		else:
			self.endog1 = self.endog
			if "excludeconstant" in kwargs.keys() and kwargs.get("excludeconstant"):
				self.exog1 = self.exog 
			else:
				self.exog1 = np.concatenate([np.ones((self.exog.shape[0], 1)), self.exog], axis=1) 
		
		return self.fit_()

	def fit_(self):
		self._initialize()
		it = 0
		s = ne.evaluate("u - x", local_dict =  {"u":self.u, "x":self.p})
		y = sp.linalg.lstsq(self.A.T, self.c)[0]
		r = ne.evaluate("c - chat", local_dict = {'c':self.c, 'chat':np.dot(self.A.T, y)})
		z = np.maximum(r, 0)
		w = ne.evaluate("z - r", local_dict = {"z":z, "r":r})
		gap = lambda c,x,b,y,u,w: ne.evaluate("cx - by + uw", local_dict = {"cx":np.dot(c, x), "by":np.dot(b, y), "uw":np.dot(u, w)})

		def stepsize(n, x, dx, s, ds, z, dz, w, dw):
			# Note: choose takes element from second vector if condition true, from first if condition false
			delta_p_lhs = np.choose(dx < 0, [np.repeat(self.big, n), ne.evaluate("-x/dx", local_dict = {"x":x, "dx":dx})])
			delta_p_rhs = np.choose(ds < 0, [np.repeat(self.big, n), ne.evaluate("-s/ds", local_dict = {"s":s, "ds":ds})])
			delta_p = min(self.beta*np.min([delta_p_lhs, delta_p_rhs]), 1)
			delta_d_lhs = np.choose(dz < 0, [np.repeat(self.big, n), ne.evaluate("-z/dz", local_dict = {"z":z, "dz":dz})])
			delta_d_rhs = np.choose(dw < 0, [np.repeat(self.big, n), ne.evaluate("-w/dw", local_dict = {"w":w, "dw":dw})])
			delta_d = min(self.beta*np.min([delta_d_lhs, delta_d_rhs]), 1)
			return delta_p, delta_d

		while gap(self.c, self.p, self.b, y, self.u, w).item() > self.eps and it < self.maxit:
			it += 1
			q = ne.evaluate("1/(z/x + w/s)", local_dict ={"z":z, "x":self.p, "w":w, "s":s})
			r = ne.evaluate("z - w", local_dict = {'z':z, 'w':w})
			rhs = np.dot(q*r, self.A.T)
			lhs = np.dot(ne.evaluate("A*q", local_dict = {"A":self.A, "q":q}), self.A.T)

			dy = np.linalg.solve(lhs, rhs)
			dx = ne.evaluate("q * (dyA - r)", local_dict = {"q": q, "dyA": np.dot(dy, self.A), "r":r})
			ds = -dx
			dz = ne.evaluate("(-z * dx) / x  - z", local_dict = {"z": z, "dx": dx, "x":self.p})
			dw = ne.evaluate("(-w * ds) / s  - w", local_dict = {"w": w, "ds": ds, "s":s})

			delta_p, delta_d = stepsize(self.c.shape[0], self.p, dx, s, ds, z, dz, w, dw)

			if (min(delta_d, delta_p) < 1):
				mu = ne.evaluate("xz + sw", local_dict = {"xz":np.dot(self.p, z.T), "sw":np.dot(s, w.T)}).item() 
				g = np.dot(ne.evaluate("x + dp*dx", local_dict = {"x":self.p, "dp":delta_p, "dx":dx}), \
						ne.evaluate("z + dd*dz", local_dict = {"z": z, "dd":delta_d, "dz":dz})) \
					+ \
					np.dot(ne.evaluate("s + dp*ds", local_dict = {"dp":delta_p, "s":s, "ds":ds}), \
						ne.evaluate("w + dd*dw", local_dict = {"dd":delta_d, "w":w, "dw":dw}))
				mu = (g/mu)**3 * (mu/(2*self.c.shape[0]))	

				dxdz = ne.evaluate("dx*dz", local_dict = {"dx": dx, "dz": dz})
				dsdw = ne.evaluate("ds*dw", local_dict = {"ds": ds, "dw": dw})
				xinv = ne.evaluate("1/x", local_dict = {"x": self.p})
				sinv = ne.evaluate("1/s", local_dict = {"s": s})
				xi = ne.evaluate("mu * (xinv-sinv)", local_dict = {"mu": mu, "xinv":xinv, "sinv": sinv})
				rhs = ne.evaluate("rhs + dot", local_dict = {"rhs":rhs, "dot": \
					np.dot(ne.evaluate("q * ((dxdz - dsdw) - xi)", local_dict = {"q":q, "dxdz": dxdz, "dsdw":dsdw, "xi":xi}), self.A.T)})

				dy = np.linalg.solve(lhs, rhs)
				dx = ne.evaluate("q * (dyA + xi- r - (dxdz - dsdw))", \
					local_dict = {"q": q, "dyA": np.dot(dy, self.A), "r":r, "xi":xi, "dxdz":dxdz, "dsdw":dsdw})
				ds = -dx
				dz = ne.evaluate("mu * xinv - z - xinv * z * dx - dxdz", local_dict = {"mu":mu, "xinv":xinv, "z":z, "dx":dx, "dxdz":dxdz})
				dw = ne.evaluate("mu * sinv - w - sinv * w * ds - dsdw", local_dict = {"mu":mu, "sinv":sinv, "w":w, "ds":ds, "dsdw":dsdw})
			
				delta_p, delta_d = stepsize(self.c.shape[0], self.p, dx, s, ds, z, dz, w, dw)
			
			self.p = ne.evaluate("x + dp * dx", local_dict = {"x":self.p, "dp":delta_p, "dx":dx})
			s = ne.evaluate("s + dp * ds", local_dict = {"s":s, "dp":delta_p, "ds":ds})
			y = ne.evaluate("y + dd * dy", local_dict = {"y":y, "dd":delta_d, "dy":dy})
			w = ne.evaluate("w + dd * dw", local_dict = {"w":w, "dd":delta_d, "dw":dw})
			z = ne.evaluate("z + dd * dz", local_dict = {"z":z, "dd":delta_d, "dz":dz})
						
		if self.isnonparametric: # TODO: check why in both cases parameters are negative
			self.params = -np.concatenate([[y[0]], y[1:].T/self.bw])
			self.normalized_cov_params = self.calculate_vcov(self.x0) 
			#print(self.x0, self.params, self.predict(self.params, self.x0), np.mean(self.endog))
		else:
			self.params = -y.T
			self.normalized_cov_params =self.calculate_vcov() 

		res = RQResults(self, self.params, normalized_cov_params=self.normalized_cov_params)
		res.fit_history['tau'] = self.tau 
		res.fit_history['iterations'] = it 
		return RQResultsWrapper(res)

	def estimatedensities(self, weighted = False):		
		# TODO: for weighted = True include intercepts
		exog = self.exog1 if weighted else self.exog
		endog = self.endog1 if weighted else self.endog
		self._exogdens = sm.nonparametric.KDEMultivariate(data=exog, var_type='c'*self.exog.shape[1], bw='normal_reference')
		self._endogdens = sm.nonparametric.KDEMultivariateConditional(endog, exog, 'c', 'c'*self.exog.shape[1], bw='normal_reference')
		#self._residdens = sm.nonparametric.KDEUnivariate(endog - np.dot(exog, self.params), 'c', bw='normal_reference')
		#self._residdens.fit()

	def predict(self, params, exog=None):
		if exog is None:
			exog = self.exog1[:,1:]
		
		return params[0] + np.dot(exog, params[1:])

	def calculate_vcov(self, x0): # based on varcov, write new function for density based
		exog = self.exog1
		endog = self.endog1
		exog = self.A.T
		endog = -self.c
		N = exog.shape[0]
		H_N = N * (self.bw) ** exog.shape[1]
		resid = endog - np.dot(exog, self.params)
		#h = (np.percentile(resid, 75) - np.percentile(resid, 25)) * N**(-2/5)
		h = np.std(resid) * N**(-2/5)
		Dinv = (N*h) * np.linalg.inv(np.dot(np.multiply(sp.stats.distributions.norm.pdf(resid/h).reshape((-1,1)), exog).T, exog))
		Omega = 1/N * np.dot(exog.T, exog)
		vcov = 1/N * self.tau * (1 - self.tau) * Dinv * Omega * Dinv
		self.PostVarianceCalculation.Fire({"H_N":h, "B1":None, "sparsity":None, "density": None, "stderrs": np.sqrt(np.diagonal(vcov))})
		return vcov
	
	def calculate_vcov_2(self, x0):
		#self.estimatedensities(weighted = True)
		#self.PostVarianceCalculation.Fire({"H_N":H_N, "B1":np.diag(self.kernel.B1), "sparsity":(1/(fyx**2)) , "density": fx, "stderrs": np.sqrt(np.diagonal(vcov))})
		pass

class RQResults(base.LikelihoodModelResults):
	fit_history = {}

	def __init__(self, model, params, normalized_cov_params, scale = 1):
		super(RQResults, self).__init__(model, params, normalized_cov_params, scale)
		self.nobs = model.nobs
		self.df_model = model.df_model
		self.df_resid = model.df_resid

	@cache_readonly
	def fittedvalues(self):
		return np.dot(self.model.exog1, self.params)

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
			('Tau:', ["%.3f" % self.fit_history['tau']]), 
			('Iterations:', ["%d" % self.fit_history['iterations']]) ]

		if not title is None:
			title = "Quantile Regression Results"
	
		from statsmodels.iolib.summary import Summary
		smry = Summary()
		smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
		smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=False)
		
		return smry


class RQResultsWrapper(lm.RegressionResultsWrapper):
	pass

wrap.populate_wrapper(RQResultsWrapper, RQResults)	

def main_2():
	data = sm.datasets.anes96.load()
	data.exog = sm.add_constant(data.exog, prepend=False)
	print(sm.OLS(data.endog, data.exog).fit().summary(xname = data.exog_name, yname = data.endog_name))
	qrresults = quantreg(data.endog, data.exog, tau=0.5).fit(excludeconstant = True) 
	print(qrresults.summary(xname=data.exog_name, yname=data.endog_name))

def main():
	import pylab as plot
	def grid1d(y, x, tau, h, size):
		parlen1 = x.shape[1] + 1
		parlen2 = x.shape[1] * (x.shape[1] + 1) + 1
		from etrics.Utilities import TriangularKernel2
		from etrics.NLRQSystem import UCIConstant
		k = TriangularKernel2()
		k.SetSigmas(np.std(x, axis=0))
		
		nlrqmodel = quantreg(y, x, tau=tau)
		for gp in np.linspace(np.min(x), np.max(x), num=size):
			dist = np.sum(np.abs(x-gp)**2,axis=1)**.5
			nlrqresults = nlrqmodel.fit(x0 = gp, kernel = k, dist = dist, bw = h) 
			yield np.concatenate([[gp], nlrqresults.params]).tolist() 

	result = {} 
	dosomethingaboutit = False
	gridpoints = 25 
	bandwidth = .2
	taus = [.1, .5, .9]
	c = dict(zip(taus, ['b', 'r', 'g']))

	N = 600
	class data:
		exog = sp.stats.distributions.uniform.rvs(0, 4*sp.pi, N)
		endog = sp.sin(exog) + sp.stats.distributions.norm.rvs(0, 0.4, N) * (exog**0.5)
		exog = exog.reshape(N, 1)
	
	for tau in taus:
		result[tau] = np.array(list(grid1d(data.endog, data.exog, tau, bandwidth, gridpoints)))

	fig=plot.figure(1, figsize=(9,13))
	plot.subplot(211)
	plot.plot(data.exog, data.endog, 'o')
	plot.grid(True)	
	for tau in taus:
		plot.plot(result[tau][:,0], result[tau][:,1], '-', c=c[tau])
		#plot.plot(result[tau][:,0], result[tau][:,1] + result[tau][:,3], '--', c=c[tau])
		#plot.plot(result[tau][:,0], result[tau][:,1] - result[tau][:,3], '--', c=c[tau])
	plot.subplot(212)	
	plot.grid(True)	
	for tau in taus:
		plot.plot(result[tau][:,0], result[tau][:,2], '-', c=c[tau])

	fig.savefig('sin.pdf', dpi=fig.dpi, orientation='portrait', bbox_inches='tight', papertype='a4')
	plot.show()

if __name__ == '__main__':
	main_2()
