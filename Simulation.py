from Utilities import EventHook, enum
from numpy import mean, var, matrix, power, multiply, double, asarray, percentile 
from scipy.stats import norm, scoreatpercentile 
from scipy import sqrt
from fractions import Fraction

Results = enum('Original', 'Bias')

class Simulation:
	__data = []
	__parameters = []
	__estimates = []
	__samplesize = []
	__evaluations = {Results.Bias:{}, Results.Original:{}} 

	def __init__(self):
		self.Generating = EventHook(maxhandlers=1)
		self.Estimating = EventHook(maxhandlers=1)
		self.PostEstimation = EventHook()

		self.AddStatistics({"Average":mean, "Variance":var}, type=Results.Original)
		self.AddStatistics({"Bias":mean}, type=Results.Bias)
		self.AddStatistics({"RMSE": lambda x,axis: sqrt(power(mean(x, axis=axis), 2)+var(x, axis=axis))})
	
	def SetParameters(self, pars, names):
		self.__parameters = pars
		self.__parnames = names

	def SetSampleSize(self, N):
		self.__samplesize = N

	def Simulate(self, S):
		for i in range(S):
			self.__data = self.Generating.Fire(self.__samplesize, self.__parameters)[0]
			self.__estimates.append(self.Estimating.Fire(self.__data)[0])
			self.PostEstimation.Fire(Fraction(i,S))
	
	def AddStatistics(self, delegates, type=Results.Bias):
		for k,v in delegates.iteritems(): 
			(self.__evaluations[type])[k] = v

	def GetResults(self, type):
		return [ (title, fct(matrix([coef -multiply(self.__parameters, double(type==Results.Bias)) \
			for coef in self.__estimates]), 0)) for title,fct in self.__evaluations[type].iteritems()] 

	def PrintTable(self):
		print "\n{0}{1}".format(" "*21, "".join(["{: >10}".format(cname) for cname in self.__parnames]))
		for k,v in self.GetResults(Results.Original)+self.GetResults(Results.Bias):
			print "{0: <20} {1}".format(k, "".join(["{:10.4f}".format(a) for a in (asarray(v).tolist())[0]]))
			

# usage example

def createData(N, theta):
	X = matrix([repeat(1, N), norm.rvs(loc=4, scale=2, size=N), norm.rvs(loc=5, scale=1, size=N)])
	eps = matrix(norm.rvs(loc=0, scale=1, size=N))
	return [X.T * matrix(theta).T + eps.T, X.T]

def estimateModel(data):
	res = models.OLS(data[0], data[1]).fit()
	return res.params

def onPostEstimation(progress):
	if progress % Fraction(1,10) == 0: print(str(progress)+"..."), 

def main():
	x = Simulation()
	x.SetParameters([1,2,3], ("beta{} "*3).format(1,2,3).split())
	x.SetSampleSize(1000)
	x.Generating += createData
	x.Estimating += estimateModel
	x.PostEstimation += onPostEstimation
	
	x.Simulate(100)

	x.AddStatistics({"95th Quantile": \
		lambda x,axis: [scoreatpercentile(x, 95, axis=axis)]}, type=Results.Original)
	x.AddStatistics({"Median": lambda x,axis: [percentile(x, 50, axis=axis)]}, type=Results.Original)

	res1 = x.GetResults(Results.Original)
	res2 = x.GetResults(Results.Bias)

	x.PrintTable()

if __name__ == '__main__':
	# main()
	pass
