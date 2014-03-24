import numpy as np 
from scipy.stats.distributions import triang

class MaxHandlerException(Exception):
	def __init__(self, maxh):
		self.expr = "MaxHandlerException"
		self.message = "You may only add up to " + maxh + " event-handling delegates"

class EventHook(object):

	def __init__(self, maxhandlers=128):
		self.__handlers = []
		self.__maxhandlers = maxhandlers

	def __iadd__(self, handler):
		if len(self.__handlers) < self.__maxhandlers:
			self.__handlers.append(handler)
		else:
			raise MaxHandlerException(self.__maxhandlers)			
		return self

	def __isub__(self, handler):
		self.__handlers.remove(handler)
		return self

	def Fire(self, *args, **keywargs):
		retvals = []
		for handler in self.__handlers:
			retvals.append(handler(*args, **keywargs))
		return retvals	

	def ClearObjectHandlers(self, inObject):
		for theHandler in self.__handlers:
			if theHandler.im_self == inObject:
				self -= theHandler

	def SetMaxHandlers(self, maxh):
		self.__maxhandlers = maxh
	
	@property
	def NumHandlers(self):
		return len(self.__handlers)

def enum(*sequential, **named):
	enums = dict(zip(sequential, range(len(sequential))), **named)
	return type('Enum', (), enums)

class Timing:
	def __init__(self, name, trc):
		self.name = name
		self.trc = trc

	def __enter__(self):
		import time
		self.start = time.time()

	def __exit__(self, type_, value, traceback):
		import time, datetime
		self.stop = time.time()
		if self.trc:
			print("{0}: {1}".format(self.name, str(datetime.timedelta(seconds=self.stop-self.start))))

def cartesian(arrays, out=None):
	arrays = [np.asarray(x) for x in arrays]
	dtype = arrays[0].dtype

	n = np.prod([x.size for x in arrays])
	if out is None:
		out = np.zeros([n, len(arrays)], dtype=dtype)

	m = n / arrays[0].size
	out[:,0] = np.repeat(arrays[0], m)
	if arrays[1:]:
		cartesian(arrays[1:], out=out[0:m,1:])
	for j in range(1, arrays[0].size):
		out[j*m:(j+1)*m,1:] = out[0:m,1:]
	return out
	
def vech(M):
	return np.array(np.matrix(M)[np.tril_indices(M.shape[0])])[0]

def varname(p):
	import inspect, re	
	for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
		m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
		if m: return m.group(1)

class multivnormal:
	def __init__(self, mean, variance):
		self._mean = mean
		self._variance = variance
	
	def rvs(self, n):
		return np.random.multivariate_normal(self._mean, self._variance, n)

	def ppf(self, tau):	
		from scipy.stats.distributions import norm
		return [norm(self._mean[i], self._variance[i][i]).ppf(tau) for i in range(len(self._mean))]

	@property
	def marginals(self):	
		from scipy.stats.distributions import norm
		return [norm(self._mean[i], self._variance[i][i]) for i in range(len(self._mean))]
	
class ConditionalDistributionWrapper:
	def __init__(self, basemarginals, inv, dinv):
		self.inv = inv
		self.dinv = dinv
		self.basemarginals = basemarginals
	
	def pdfjoint(self, x0, y): 
		return np.prod(self.basepdfs(x0, y)) * np.abs(np.linalg.det(self.dinv(y, x0)))

	def pdf(self, y, x0):
		# TODO does that work like this with the marginals - if at all works ONLY if errors enter diagonally
		# print(self.basemarginals[0].pdf(.1), self.basemarginals[1].pdf(.05))
		# return self.basepdfs(x0, y) * np.abs(np.diagonal(self.dinv(y, x0))).T
		# this should work due to independence of epsilons, f_x1 = \int f(x1) f(x2) dx2 = J f_e1(xinv)
		# self.plotjoint(x0)
		# WARNING: this works only if epsilons enter diagonally, otherwise we need to obtain marginals from joint
		return self.basepdfs(x0, y) * np.abs(np.diagonal(self.dinv(y, x0)))

	def plotjoint(self, x0):
		#import matplotlib.pyplot as plt
		#from matplotlib import cm
		#from mpl_toolkits.mplot3d.axes3d import Axes3D
		#from etrics.Utilities import matrixbyelem
		#X, Y = np.meshgrid(np.linspace(20,36,100), np.linspace(10,26,100))
		#l = lambda y1, y2: self.pdfjoint(x0, np.array([y1, y2]))
		#Z = matrixbyelem(X, Y, l)
		#fig, ax = plt.subplots()
		#p = ax.pcolor(X, Y, Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), linewidth=0)
		#cb = fig.colorbar(p, ax=ax)
		#plt.show()
		from etrics.Utilities import quickjointdensityplot
		quickjointdensityplot(np.array([16,36]), np.array([10,26]), lambda y1, y2: self.pdfjoint(x0, np.array([y1, y2])))
		
	def basepdfs(self, x0, y):
		return np.array([f.pdf(e) for f,e in zip(self.basemarginals, self.inv(y, x0))])

class TriangularKernel:
	def __init__(self, scale):
		self.c = 0.5
		self.loc = -scale/2 
		self.scale = scale
		
	def pdf(self, y):
		return triang.pdf(y, self.c, loc = self.loc, scale = self.scale)

	def moment(self, n):
		return triang.moment(n, self.c, loc = self.loc, scale = self.scale)
	
	def moment2(self, n):
		return [1.0, 1.+ 4/3 + 1/2, (1/3 + 1/2 + 1/5)*self.scale][n]
	
	def momentprime2(self, n):
		return (np.array([16 , 4, 4/3]) * (self.scale ** np.array([-3,-2,-1])))[n] 

def quickdensityplot( data):	
	import pylab as plt
	from numpy import min, max
	import statsmodels.api as sm
	x = np.linspace(min(data), max(data))
	e = sm.nonparametric.KDEUnivariate(data)
	e.fit()
	plt.plot(x, e.evaluate(x))
	plt.show()

def quickjointdensityplot(y1, y2, dens = None):
	import matplotlib.pyplot as plt
	import statsmodels.api as sm
	from matplotlib import cm
	from mpl_toolkits.mplot3d.axes3d import Axes3D
	from etrics.Utilities import matrixbyelem
	X, Y = np.meshgrid(np.linspace(np.min(y1),np.max(y1), 100), np.linspace(np.min(y2), np.max(y2),100))
	if dens is None:
		est = sm.nonparametric.KDEMultivariate(data=np.hstack([y1, y2]), var_type='cc', bw='normal_reference')
		dens = lambda y1, y2: est.pdf(np.array([y1, y2])) 
	dens(X[0,0], Y[0,0])	

	Z = matrixbyelem(X, Y, dens)
	fig, ax = plt.subplots()
	p = ax.pcolor(X, Y, Z, cmap=cm.RdBu, vmin=abs(Z).min(), vmax=abs(Z).max(), linewidth=0)
	cb = fig.colorbar(p, ax=ax)
	plt.show()
		
	def basepdfs(self, x0, y):
		return np.array([f.pdf(e) for f,e in zip(self.basemarginals, self.inv(y, x0))])


def matrixbyelem(x, y, f):
	M =  np.ones(x.shape)
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			M[i,j] = f(x[i,j], y[i,j])	
	return M			

class EstimatedDistWrapper:	
	def __init__(self, est):
		self.e = est
	def pdf(self, x):
		return self.e.evaluate(x).item()

