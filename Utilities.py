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

def enum(*sequential, **named):
	enums = dict(zip(sequential, range(len(sequential))), **named)
	return type('Enum', (), enums)

class Timing:
	def __init__(self, name):
		self.name = name

	def __enter__(self):
		import time
		self.start = time.time()

	def __exit__(self, type_, value, traceback):
		import time
		self.stop = time.time()
		print("Time for {0}: {1:.3f} ms".format(self.name, (self.stop-self.start)*1000.))
		return True
