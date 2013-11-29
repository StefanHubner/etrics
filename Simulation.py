from etrics.Utilities import EventHook, enum
from numpy import repeat,mean, var, std,matrix, power, multiply, double, asarray, percentile 
from scipy.stats import norm, scoreatpercentile 
from scipy import sqrt
from fractions import Fraction

Results = enum('Original', 'Bias')

class TryAgain(Exception):
	def __init__(self, expr, msg):
		self.expr = expr
		self.msg = msg
	def __str__(self):
		return repr(self.value)

class Simulation:
	__data = []
	__parameters = []
	__estimates = []
	__samplesize = []
	__evaluations = {Results.Bias:{}, Results.Original:{}} 
	__estimationpars = {}
	__auxiliaryparameters = {};	

	def __init__(self):
		self.Generating = EventHook(maxhandlers=1)
		self.Estimating = EventHook(maxhandlers=1)
		self.PreEstimation = EventHook()
		self.Warning = EventHook()

		self.AddStatistics({"Average":mean, "Std.Dev.":std}, type=Results.Original)
		self.AddStatistics({"Bias":mean}, type=Results.Bias)
		self.AddStatistics({"RMSE": lambda x,axis: sqrt(power(mean(x, axis=axis), 2)+var(x, axis=axis))})
	
	def SetIdentifiedParameters(self, pars, names):
		self.__parameters = pars
		self.__parnames = names
	
	def SetStructuralParameters(self, pars):
		self.__structuralpars = pars

	def SetParameters(self, pars, names):
		self.SetIdentifiedParameters(pars, names)
		self.SetStructuralParameters(pars)

	def SetEstimationParameters(self, **kwargs):
		self.__estimationpars = kwargs
	
	def SetSamplingParameters(self, **kwargs):
		self.__samplingpars = kwargs

	def Simulate(self, S):
		for i in range(S):
			self.PreEstimation.Fire(Fraction(i,S))
			success = False
			while not success:
				self.__data = self.Generating.Fire(self.__structuralpars, **self.__samplingpars)[0]
				try:
					est = self.Estimating.Fire(self.__data, **self.__estimationpars)[0]
					self.__estimates.append(est)
				except TryAgain as ta:
					self.Warning.Fire(ta)
					success = False
				else:
					success = True
				
	
	def AddStatistics(self, delegates, type=Results.Bias):
		for k,v in delegates.items(): 
			(self.__evaluations[type])[k] = v

	def GetResults(self, type):
		return [ (title, fct(matrix([coef -multiply(self.__parameters, double(type==Results.Bias)) \
			for coef in self.__estimates]), 0)) for title,fct in self.__evaluations[type].items()] 

	def PrintTable(self):
		print ("\n{0}{1}".format(" "*21, "".join(["{: >10}".format(cname) for cname in self.__parnames])))
		for k,v in self.GetResults(Results.Original)+self.GetResults(Results.Bias):
			print ("{0: <20} {1}".format(k, "".join(["{:10.4f}".format(a) for a in (asarray(v).tolist())[0]]))) 

# usage example

def createData(theta, N):
	#N = samplingpars["N"]
	X = matrix([repeat(1, N), norm.rvs(loc=4, scale=2, size=N), norm.rvs(loc=5, scale=1, size=N)])
	eps = matrix(norm.rvs(loc=0, scale=1, size=N))
	return [X.T * matrix(theta).T + eps.T, X.T]

def estimateModel(data):
	import statsmodels.regression.linear_model as model
	res = model.OLS(data[0], data[1]).fit()
	return res.params

def Progress(progress):
	if progress % Fraction(1,10) == 0: 
		print(str(progress)+"...") 

def onWarning(ex):
	print("warning: DGP caused estimation to fail: " + ex.msg)

def main():
	x = Simulation()
	x.SetParameters([1,2,3], ("beta{} "*3).format(1,2,3).split())
	x.SetSamplingParameters(N=1000)
	x.Generating += createData
	x.Estimating += estimateModel
	x.PreEstimation += Progress 
	x.Warning += onWarning
	
	x.Simulate(100)

	x.AddStatistics({"95th Quantile": \
		lambda x,axis: [scoreatpercentile(x, 95, axis=axis)]}, type=Results.Original)
	x.AddStatistics({"Median": lambda x,axis: [percentile(x, 50, axis=axis)]}, type=Results.Original)

	res1 = x.GetResults(Results.Original)
	res2 = x.GetResults(Results.Bias)

	x.PrintTable()

if __name__ == '__main__':
	main()
	# pass
