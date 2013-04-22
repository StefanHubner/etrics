from Utilities import EventHook, enum
from numpy import * 
from scipy.stats import * 
import statsmodels.api as models

Results = enum('Original', 'Bias')

class Simulation:
	__data = []
	__parameters = []
	__estimates = []
	__samplesize = []

	def __init__(self):
		self.Generating = EventHook(maxhandlers=1)
		self.Estimating = EventHook(maxhandlers=1)
		self.PostEstimation = EventHook()
	
	def SetParameters(self, pars):
		self.__parameters = pars

	def SetSampleSize(self, N):
		self.__samplesize = N

	def Simulate(self, S):
		for i in range(S):
			self.__data = self.Generating.Fire(self.__samplesize, self.__parameters)[0]
			self.__estimates.append(self.Estimating.Fire(self.__data)[0])
	
	def GetResults(self, delegates, type=Results.Bias):
		return [ (title, delegates[title](matrix([coef -multiply(self.__parameters, double(type==Results.Bias)) \
			for coef in self.__estimates]), 0)) for title in delegates]

# usage

def createData(N, theta):
	X = matrix([repeat(1, N), norm.rvs(4, 2, N), norm.rvs(5, 1, N)])
	eps = matrix(norm.rvs(0, 1, N))
	return [X.T * matrix(theta).T + eps.T, X.T]

def estimateModel(data):
	res = models.OLS(data[0], data[1]).fit()
	return res.params

x = Simulation()
x.SetParameters([1,2,3])
x.SetSampleSize(1000)
x.Generating += createData
x.Estimating += estimateModel
x.Simulate(100)
res1 = x.GetResults( {"Bias":mean, "RMSE": lambda x,axis: sqrt(power(mean(x, axis=axis), 2)+var(x, axis=axis))}, \
	type=Results.Bias )
res2 = x.GetResults( {"Average":mean, "Variance":var, \
	"95th Quantile": lambda x,axis: [scoreatpercentile(x,95, axis=axis)]}, type=Results.Original )

for k,v in res1+res2:
	print "{0: <20} {1}".format(k, (" "*5).join(["{: .4f}".format(a) for a in (asarray(v).tolist())[0]]))
