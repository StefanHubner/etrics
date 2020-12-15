from etrics.Utilities import EventHook
from scipy.optimize import minimize
from scipy import stats

from statsmodels.tools.decorators import (resettable_cache, cache_readonly, cache_writable)
from numpy.linalg import matrix_rank

import numpy as np
import statsmodels.api as sm
import statsmodels.base.model as base
import statsmodels.base.wrapper as wrap
import statsmodels.regression.linear_model as lm

class RQ(base.LikelihoodModel):
	""" 
	min_x {c'x | Ax = b, 0 <= x <= u }

	c = -y 

	equality constraint (part of KKR matrix)
		Aeq = X' 
		beq=(1-tau)*iota_N.T*X.T
	
	inequality constraints: define this as A and b
		u=iota_N.T
		[ id_N] *a <= [u]
		[-id_N]    <= [0]
	"""	

	def __init__(self, endog, exog, **kwargs):
		super(RQ, self).__init__(endog, exog, **kwargs)
		self._initialize()
		self.PostEstimation = EventHook()

	def _initialize(self):
		self.nobs = float(self.endog.shape[0])
		self.df_resid = np.float(self.exog.shape[0] - matrix_rank(self.exog))
		self.df_model = np.float(matrix_rank(self.exog)-1)
		
		self.c = -self.endog
		self.A = np.concatenate([np.identity(self.endog.shape[0]), -np.identity(self.endog.shape[0])], axis=0)
		self.Aeq = self.exog.T
		self.b = np.concatenate([np.ones(self.endog.shape[0]), np.zeros(self.endog.shape[0])], axis=0)
		self.beq = (1-self.tau) * sum(self.exog, 0)
		self.t = 1 
		self.eps = 10e-07
		self.maxit = 1
		self.update = 1.1
	
	def fit(self, **kwargs):

		a = (1-self.tau) * np.ones(self.exog.shape[0]) # start on the boundary
		iteration = 0
		while self.b.shape[0]/self.t > self.eps and iteration < self.maxit:
			r = minimize(self.lpip, a, method='Newton-CG', jac=self.grad_lpip, hess=self.hess_lpip,
				options={'disp': True})
			a = r.x	
			self.t = self.update**iteration 
			iteration += 1

		self.normalized_cov_params =  np.identity(self.exog.shape[1])
		res = RQResults(self, np.ones(self.exog.shape[1]), normalized_cov_params=self.normalized_cov_params)
		res.fit_history['outer_iterations'] = 10
		res.fit_history['avg_inner_iterations'] = 5.2
		return RQResultsWrapper(res)

	def predict(self, params, exog=None):
		if exog is None:
			exog = self.exog
		
		return np.dot(exog, params)

	def lpip(self, par): 
		s = self.b - np.dot(self.A, par)
		return np.dot(self.c.T, par) * self.t + sum(-np.log(s), 1) if min(s)>0 else np.inf

	def grad_lpip(self, par):
		# zeros for rest of equality components, don't move away from there
		return self.c * self.t + np.dot(self.A.T, np.divide(1., self.b - np.dot(self.A, par)))

	def hess_lpip(self, par): 
		# TODO: use sparsity of this matrix
		# equality constraints enter the hessian in KKR matrix form (see HOptimize) (H A' | A 0)
		return np.dot(np.dot(self.A.T, np.diag( np.divide(1., np.power(self.b - np.dot(self.A, par), 2)) )), self.A)

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
	xname = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"]
	yname = ["y"]
	qrresults = RQ(data.endog, data.exog, tau=0.9).fit() 
	print(qrresults.summary(xname=xname, yname=yname))

if __name__ == '__main__':
	main()
