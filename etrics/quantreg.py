from etrics.Utilities import EventHook
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
	min_x {c'x | Ax = b, 0 <= x <= u }
	"""	

	def __init__(self, endog, exog, **kwargs):
		super(quantreg, self).__init__(endog, exog, **kwargs)
		self._initialize()
		self.PostEstimation = EventHook()

	def _initialize(self):
		self.nobs = float(self.endog.shape[0])
		self.df_resid = np.float(self.exog.shape[0] - rank(self.exog))
		self.df_model = np.float(rank(self.exog)-1)
		
		self.c = -self.endog
		self.A = self.exog.T
		self.b = (1-self.tau) * sum(self.exog, 0)
		self.u = np.ones(self.nobs)
		self.p  = (1-self.tau) * self.u # start on boundary
		self.beta = 0.9995 
		self.eps = 10e-05
		self.maxit = 50 
		self.big = 10e+20
	
	def fit(self, **kwargs):
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

			delta_p, delta_d = stepsize(self.nobs, self.p, dx, s, ds, z, dz, w, dw)

			if (min(delta_d, delta_p) < 1):
				mu = ne.evaluate("xz + sw", local_dict = {"xz":np.dot(self.p, z.T), "sw":np.dot(s, w.T)}).item() 
				g = np.dot(ne.evaluate("x + dp*dx", local_dict = {"x":self.p, "dp":delta_p, "dx":dx}), \
						ne.evaluate("z + dd*dz", local_dict = {"z": z, "dd":delta_d, "dz":dz})) \
					+ \
					np.dot(ne.evaluate("s + dp*ds", local_dict = {"dp":delta_p, "s":s, "ds":ds}), \
						ne.evaluate("w + dd*dw", local_dict = {"dd":delta_d, "w":w, "dw":dw}))
				mu = (g/mu)**3 * (mu/(2*self.nobs))	

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
			
				delta_p, delta_d = stepsize(self.nobs, self.p, dx, s, ds, z, dz, w, dw)
			
			self.p = ne.evaluate("x + dp * dx", local_dict = {"x":self.p, "dp":delta_p, "dx":dx})
			s = ne.evaluate("s + dp * ds", local_dict = {"s":s, "dp":delta_p, "ds":ds})
			y = ne.evaluate("y + dd * dy", local_dict = {"y":y, "dd":delta_d, "dy":dy})
			w = ne.evaluate("w + dd * dw", local_dict = {"w":w, "dd":delta_d, "dw":dw})
			z = ne.evaluate("z + dd * dz", local_dict = {"z":z, "dd":delta_d, "dz":dz})
			
						
		self.params = y.T 
		self.normalized_cov_params =  np.identity(self.exog.shape[1])

		res = RQResults(self, self.params, normalized_cov_params=self.normalized_cov_params)
		res.fit_history['outer_iterations'] = it 
		res.fit_history['avg_inner_iterations'] = 0 
		return RQResultsWrapper(res)

	def predict(self, params, exog=None):
		if exog is None:
			exog = self.exog
		
		return np.dot(exog, params)

class RQResults(base.LikelihoodModelResults):
	fit_history = {}

	def __init__(self, model, params, normalized_cov_params, scale = 1):
		super(RQResults, self).__init__(model, params, normalized_cov_params, scale)
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
			title = "Quantile Regression Results"
	
		from statsmodels.iolib.summary import Summary
		smry = Summary()
		smry.add_table_2cols(self, gleft=top_left, gright=top_right, yname=yname, xname=xname, title=title)
		smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha, use_t=False)
		
		return smry


class RQResultsWrapper(lm.RegressionResultsWrapper):
	pass

wrap.populate_wrapper(RQResultsWrapper, RQResults)	

def main():
	data = sm.datasets.longley.load()
	data.exog = sm.add_constant(data.exog, prepend=False)
	print(sm.OLS(data.endog, data.exog).fit().summary())
	xname = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
	yname = ["y"]
	qrresults = quantreg(data.endog, data.exog, tau=0.5).fit() 
	print(qrresults.summary(xname=xname, yname=yname))

if __name__ == '__main__':
	main()
