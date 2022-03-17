import numpy as np
import os
from timeit import default_timer as timer
import math
from core import *


class Sprintz(CompressionAlgorithm):
	'''Applies a basic quantization algorithm to compress the data
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.coderange = int(math.ceil(1.0/(error_thresh)))


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		codes = np.ones((self.N, self.p))*-1#set all to negative one

		for i in range(self.N):
			for j in range(self.p):
				codes[i,j] = int(self.data[i,j]*self.coderange)

		codes, decode = delta_xform(codes)

		fname = self.target + '/learned'
		np.save(fname, decode)

		self.DATA_FILES += [fname + '.npy']


		struct = iarray_bitpacking(codes)
		struct.flush(self.CODES)
		struct.flushmeta(self.METADATA)

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']
		self.compression_stats['code_size'] = self.getSize() - self.getModelSize()
		self.compression_stats['model_size'] = self.getModelSize()
		

	def decompress(self, original=None):

		start = timer()

		struct = BitPackedStruct.loadmeta(self.METADATA)
		codes = struct.load(self.CODES)

		normalization = np.load(self.NORMALIZATION + '.npy')
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])
		bit_length = struct.bit_length

		fname = self.target + '/learned.npy'
		decode = np.load(fname)
		codes = i_delta_xform(codes, decode)
		coderange = np.max(codes)

		for i in range(p):
			codes[:,i] = (codes[:,i]/coderange)*(normalization[0,i] - normalization[1,i]) + normalization[1,i]

		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes


class SprintzGzip(CompressionAlgorithm):
	'''Applies a basic quantization algorithm to compress the data
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.coderange = int(math.ceil(1.0/(error_thresh)))


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		codes = np.ones((self.N, self.p))*-1#set all to negative one

		for i in range(self.N):
			for j in range(self.p):
				codes[i,j] = int(self.data[i,j]*self.coderange)

		codes, decode = delta_xform(codes)

		fname = self.target + '/learned'
		np.save(fname, decode)

		struct = iarray_bitpacking(codes)
		struct.flushz(self.CODES)
		struct.flushmeta(self.METADATA)

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']
		self.compression_stats['code_size'] = self.getSize() - self.getModelSize()
		self.compression_stats['model_size'] = self.getModelSize()
		

	def decompress(self, original=None):

		start = timer()

		struct = BitPackedStruct.loadmeta(self.METADATA)
		codes = struct.loadz(self.CODES)


		normalization = np.load(self.NORMALIZATION + '.npy')
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])
		bit_length = struct.bit_length

		fname = self.target + '/learned'

		decode = np.load(fname+'.npy')
		codes = i_delta_xform(codes, decode)
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

"""
data = np.loadtxt('/Users/sanjaykrishnan/Downloads/HT_Sensor_UCIsubmission/HT_Sensor_dataset.dat')[:,1:]
#data = np.load('/Users/sanjaykrishnan/Downloads/ts_compression/l2c/data/electricity.npy')

#normalize this data
N,p = data.shape


nn = SprintzGzip('quantize')
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)
"""

