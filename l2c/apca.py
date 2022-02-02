import numpy as np
import os
from timeit import default_timer as timer
import math
from core import *


class AdaptivePiecewiseConstant(CompressionAlgorithm):
	'''Applies a quantization + entropy coding to 
	   compress a dataset similar to Squish.

	   Assumes time is along columns
	'''

	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		codes = np.zeros((self.N, self.p))*-1#set all to negative one

		for j in range(self.p):
			for i in range(self.N):

				if i == 0:
					codes[i,j] = self.data[i,j]
				else:
					if np.abs(codes[i-1,j] - self.data[i,j]) < self.error_thresh:
						codes[i,j] = codes[i-1,j]
					else:
						codes[i,j] = self.data[i,j]

		codes = codes.flatten(order='F')
		fname = self.CODES
		np.save(fname, codes)

		compressz(self.CODES + '.npy', self.CODES+'.npyz')
		self.DATA_FILES += [self.CODES + '.npyz']


		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']

		

	def decompress(self, original=None):

		start = timer()

		decompressz(self.CODES + '.npyz', self.CODES+'.npy')
		codes = np.load(self.CODES+".npy")

		normalization = np.load(self.NORMALIZATION + '.npy')
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])

		codes = codes.reshape(N,p, order='F').astype(np.float64)
		coderange = np.max(codes)

		for i in range(p):
			codes[:,i] = (codes[:,i])*(normalization[0,i] - normalization[1,i]) + normalization[1,i]


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes



####
"""
Test code here
"""
####

data = np.loadtxt('/Users/sanjaykrishnan/Downloads/HT_Sensor_UCIsubmission/HT_Sensor_dataset.dat')[:2000,1:]
#data = np.load('/Users/sanjaykrishnan/Downloads/ts_compression/l2c/data/electricity.npy')
print(data.shape)
#data = np.nan_to_num(data)

#normalize this data
N,p = data.shape


nn = AdaptivePiecewiseConstant('quantize')
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)


