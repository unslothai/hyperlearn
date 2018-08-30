from torch.multiprocessing import Pool, cpu_count
from .base import isIterable, isTensor
from copy import copy

__all__ = ['ParallelReference']


class ParallelReferenceError(BaseException):
	def __init__(self, text = 'At least 1 item needs to be type (array, list), '
								'as that acts as the base item for which the '
								'multiprocessing functions.'):
		self.text = text
	def __str__(self):
		return self.text

"""
------------------------------------------------------------
Parallel_Reference
	This module requires a list / vector to act as a "reference"

Updated 27/8/2018
------------------------------------------------------------
"""
class ParallelReference(BaseEstimator):
	'''
	Parallel_Reference's syntax is like joblib.
	Parallel_Reference(f, n_jobs = 1, reference = -1)

	Eg: 
		Parallel_Reference(f)(X, [1,2,3], 1)

		[1,2,3] (index 1) acts as the reference, so f(X, 1), f(X, 2), ...
		is called.

	If you want instead given f, X, just process f(X[0:10]), f(X[10:20]), ...
	maybe use ParallelProcess instead.
	'''
	
	def __init__(self, f, n_jobs = 1, reference = -1):
		
		self.count = cpu_count()
		assert type(n_jobs) is int
		assert type(reference) is int
		
		if n_jobs == -1 or n_jobs > self.count:
			self.n_jobs = self.count
		elif n_jobs == 'fit':
			self.n_jobs = None
		else:
			self.n_jobs = n_jobs
		self.f = f
		self.reference = reference
		
		
	def __call__(self, *args, **kwargs):
		
		if self.n_jobs == 1:
			args = list(args)
			if self.reference > -1:
				output = []

				for j in copy(args[self.reference]):
					args[self.reference] = j
					out = self.f(*args)
					output.append(out)
			else:
				raise ParallelReferenceError()
		else:
			output = self.multiprocess(*args, **kwargs)

		return self.process(output)
		
	
	def multiprocess(self, *args, **kwargs):
		foundIter = False
		if self.reference > -1:
			foundIter = self.reference
			length = len(args[self.reference])
		else:
			for i,x in enumerate(args):
				if isIterable(x):
					foundIter = i
					length = len(x)
					break

		if self.n_jobs is None:
			self.n_jobs = length if length > self.count else self.count
		self.n_jobs = length if self.count > length else self.n_jobs
			
		if foundIter is False:
			raise ParallelReferenceError()
		
		toCall = []
		for i,x in enumerate(args):
			if foundIter == i:
				toCall.append(x)
				continue
			if isTensor(x):
				x.share_memory_()
			toCall.append([x]*length)
		
		with Pool(processes = self.n_jobs) as pool:
			output = pool.starmap(self.f, zip(*toCall) )
		return output


	def process(self, output):
		finalOutput = []
		for i in range(len(output[0])):
			eachVariable = []
			for j in output:
				eachVariable.append(j[i])
			finalOutput.append(eachVariable)
		return finalOutput

		