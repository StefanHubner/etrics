from etrics.Utilities import EventHook, enum
from numpy import repeat,mean, var, std,matrix, power, multiply, double, asarray, percentile 
from scipy.stats import norm, scoreatpercentile 
from scipy import sqrt
from fractions import Fraction
from collections import OrderedDict

import time, sys, types, pickle, logging, os

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
	__samplesize = []
	__evaluations = {Results.Bias:OrderedDict(), Results.Original:OrderedDict()} 
	__estpars = {}
	__auxiliaryparameters = {};	

	def __init__(self, statefile, logger = None):
		self.Generating = EventHook(maxhandlers=1)
		self.Estimating = EventHook(maxhandlers=1)
		self.PreEstimation = EventHook()
		self.PostGeneration = EventHook()
		self.PostEstimation = EventHook()
		self.Warning = EventHook()
		self.__statefile = statefile
		self.__estimates = []
		self.__actcnt = 0
		if logger is None:
			self.logger = logging.getLogger("simulation")
		else:
			self.logger = logger

		quantile = lambda x, t: percentile(x, t*100, axis=0)
		self.AddStatistics({"Mean": lambda x: mean(x, axis=0)}, type=Results.Original)
		self.AddStatistics({"Median": lambda x: quantile(x, .5)}, type=Results.Original)
		self.AddStatistics({"Std.Dev.": lambda x: std(x, axis=0)}, type=Results.Original)
		self.AddStatistics({"Mean Bias": lambda x: mean(x, axis=0)})
		self.AddStatistics({"Median Bias": lambda x: quantile(x, .5)})
		self.AddStatistics({"RMSE": lambda x: sqrt(power(mean(x, axis=0), 2)+var(x, axis=0))})
	
	def SetSeed(self, seed):
		import numpy.random
		numpy.random.seed(seed=seed)

	def SetIdentifiedParameters(self, pars, names):
		self.__parameters = pars
		self.__parnames = names
	
	def SetStructuralParameters(self, pars, functionalform):
		self.__strpars = pars
		self.__form = functionalform

	def SetParameters(self, pars, names):
		self.SetIdentifiedParameters(pars, names)
		self.SetStructuralParameters(pars)

	def SetEstimationParameters(self, **kwargs):
		self.__estpars = kwargs
	
	def SetSamplingParameters(self, **kwargs):
		self.__smplpars = kwargs
	
	def CheckState(self, resume):
		if os.path.exists(self.__statefile) and (resume or input("Statefile found. Resume? [y]es, [n]o?") == "y"):
			with open(self.__statefile, "rb") as f:
				self.__estimates = pickle.load(f)
				self.__actcnt = len(self.__estimates)

	def Simulate(self, S, resume = False):
		self.CheckState(resume)
		for i in range(self.__actcnt, S):
			self.PreEstimation.Fire(Fraction(i,S))
			starttime = time.time()
			success = False
			while not success:
				try:
					self.__data = self.Generating.Fire(self.__strpars, self.__form, **self.__smplpars)[0]
					self.PostGeneration.Fire(time.time()-starttime)		
					est = self.Estimating.Fire(self.__data, **self.__estpars)[0]
				except TryAgain as ta:
					self.Warning.Fire(ta)
					success = False
				except KeyboardInterrupt as ki:
					option = input("Simulation was interrupted: e[x]it, [p]rint, [t]hrow, [c]ontinue? ")
					if option == 'x':
						return	
					elif option == 'p':
						self.PrintTable()
					elif option == 't':
						raise(ki)
					print("Moving on...")	
					success = False
				else:
					self.__estimates.append(est)
					success = True
					self.__actcnt += 1
					self.WriteTable()
					with open("{0}/Simulation.{1}.bin".format(dir, name), "wb") as f:
						pickle.dump(self.__estimates, f)

			self.PostEstimation.Fire(time.time()-starttime)		
	
	def Load(self, dir, name):
		with open("{0}/Simulation.{1}.bin".format(dir, name), "rb") as f:
			self.__dict__ = pickle.load(f)
	
	def Save(self, dir, name):
		with open("{0}/Simulation.{1}.bin".format(dir, name), "wb") as f:
			pickle.dump(self.__dict__, f)
	
	def AddStatistics(self, delegates, type=Results.Bias):
		for k,v in delegates.items(): 
			(self.__evaluations[type])[k] = v

	def GetResults(self, type):
		return [ (title, asarray(fct(matrix([coef -multiply(self.__parameters, double(type==Results.Bias)) \
			for coef in self.__estimates]))).flatten().tolist()) for title,fct in self.__evaluations[type].items()] 
	
	def PrintTable(self, cols = 5):
		self.WriteTable("stdout", cols)

	def WriteLine(self, ident, line, always = False):
		if self.__uselatex or always:
			self.__tablefilehandle.write("{0}{1}\n".format("\t"*ident*int(self.__uselatex), line))

	def SetWritingOptions(self, filename, cols = 5, standalone = False, caption = None, label = None):
		self.__tablefilename = filename
		self.__cols = cols
		self.__standalone = standalone
		self.__caption = caption
		self.__label = label 

	def WriteTable(self):
		self.__tablefilehandle = open(self.__tablefilename, "w") if self.__tablefilename is not "stdout" else sys.stdout
		self.__uselatex = self.__tablefilehandle != sys.stdout 
		if self.__standalone: 
			self.WriteLine(0, "\\documentclass{article}\n\\usepackage{booktabs}\n\\usepackage{listings}\\lstset{basicstyle=\\scriptsize}\n")
			self.WriteLine(0, "\\usepackage[top=1cm,bottom=1cm,left=1cm,right=1cm]{geometry}\n\\usepackage{amssymb}\n\\begin{document}\n")

		self.WriteLine(0, "\\begin{table}[h!]")
		self.WriteLine(1, "\\centering")
		self.WriteLine(1, "\\begin{{tabular}}{{l|{0}}}".format("c"*self.__cols))
		self.WriteLine(1, "\\toprule")

		self.WriteTableInt([("True Values", self.__parameters)]+self.GetResults(Results.Original)+self.GetResults(Results.Bias), self.__parnames, self.__cols)

		self.WriteLine(1, "\\bottomrule")
		self.WriteLine(1, "\\end{tabular}")
		
		if self.__caption is not None: 
			self.WriteLine(1, "\\caption{{{0}}}".format(self.__caption))
		
		if self.__label is not None: 
			self.WriteLine(1, "\\label{{{0}}}".format(self.__label))
		
		self.WriteLine(0, "\\end{table}")
			
		if self.__standalone:
			self.WriteLine(0, "\\pagebreak\n\\begin{lstlisting}")
		else:
			self.WriteLine(0, "\\iffalse")

		for k,v in [("Fct.Form",self.__form),("B",self.__actcnt)]+list(self.__smplpars.items())+list(self.__estpars.items())+list(self.__strpars.items()):
			if not (isinstance(v, types.FunctionType) or (isinstance(v, list) and isinstance(v[0], types.FunctionType))):
				self.WriteLine(0, "{2!s:} {0!s: <20} = {1!s: <50}".format(k, v, "%" if not self.__standalone else ""))

		if self.__standalone: 
			self.WriteLine(0, "\\end{lstlisting}")	
			self.WriteLine(0, "\\end{document}\n")
		else:
			self.WriteLine(0, "\\fi")
		
		if self.__uselatex: self.__tablefilehandle.close()

	def WriteTableInt(self, results, names, cols):
		hnames, tnames = names[:cols], names[cols:]
		tres = []

		cdelim, rdelim = (" & ", " \\\\") if self.__uselatex else ("", "")
		maxlen = max([len(n) for n in hnames]) 

		self.WriteLine (2, "\n{0}{1}{2}".format(" "*21, "".join(["{0} {1: >{w}}".format(cdelim, cname, w=maxlen) for cname in hnames]), rdelim), True)
		self.WriteLine (1, "\\midrule")

		for k,v in results:
			tres.append((k, v[cols:]))
			self.WriteLine (2, "{0: <20} {1} {2} {3}".format(k, cdelim, cdelim.join(["{0:{w}.4f}".format(a, w=maxlen) for a in v[:cols]]), rdelim), True) 

		if len(tnames) > 0:
			self.WriteLine(1, "\\midrule")
			self.WriteTableInt(tres, tnames, cols)
		
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

	x.AddStatistics({"95th Quantile": lambda x: [scoreatpercentile(x, 95, axis=0)]}, type=Results.Original)

	res1 = x.GetResults(Results.Original)
	res2 = x.GetResults(Results.Bias)

	x.WriteTable()

if __name__ == '__main__':
	main()
	# pass
