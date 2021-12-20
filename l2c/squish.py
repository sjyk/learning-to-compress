import numpy as np
import os
from timeit import default_timer as timer
import math
from core import *


class EntropyCoding(CompressionAlgorithm):
	'''Applies a quantization + entropy coding to 
	   compress a dataset similar to Squish.
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.coderange = int(math.ceil(1.0/error_thresh))

		self.TURBO_CODE_LOCATION = "Turbo-Range-Coder/turborc" 
		self.TURBO_CODE_PARAMETER = "-25" #on my laptop run -e0 and find best solution


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		codes = np.ones((self.N, self.p))*-1#set all to negative one

		for i in range(self.N):
			for j in range(self.p):
				codes[i,j] = int(self.data[i,j]*self.coderange)

		codes = codes.astype(np.intc).flatten(order='F') #set as a c-integer type
		fname = self.CODES
		np.save(fname, codes)

		command = " ".join([self.TURBO_CODE_LOCATION, self.TURBO_CODE_PARAMETER, fname+".npy", fname+".npy"])

		os.system(command)

		self.DATA_FILES += [fname+".npy.rc"]


		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']

		

	def decompress(self, original=None):

		start = timer()

		command = " ".join([self.TURBO_CODE_LOCATION, "-d", self.CODES+".npy.rc", self.CODES+".npy"])
		os.system(command)
		codes = np.load(self.CODES+".npy")

		normalization = np.load(self.NORMALIZATION + '.npy')
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])

		codes = codes.reshape(N,p, order='F').astype(np.float64)
		coderange = np.max(codes)

		for i in range(p):
			codes[:,i] = (codes[:,i]/coderange)*(normalization[0,i] - normalization[1,i]) + normalization[1,i]


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes



####
"""
Test code here
"""
####

data = np.loadtxt('/Users/sanjaykrishnan/Downloads/HT_Sensor_UCIsubmission/HT_Sensor_dataset.dat')[:,1:]

#normalize this data
N,p = data.shape


nn = EntropyCoding('quantize')
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)


