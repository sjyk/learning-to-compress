import numpy as np
import os
from timeit import default_timer as timer
import math
import struct
from core import *


class Identity(CompressionAlgorithm):
	'''Baseline lossless compression
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.RAW = self.target + '/data'
		self.DATA_FILES += [self.RAW + '.npy']


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		np.save(self.RAW, self.data)

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']

		

	def decompress(self, original=None):

		start = timer()

		codes = np.load(self.RAW + '.npy')

		normalization = np.load(self.NORMALIZATION + '.npy')
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])

		for i in range(p):
			codes[:,i] = (codes[:,i] + normalization[1,i])*(normalization[0,i] - normalization[1,i])


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes



class IdentityGZ(CompressionAlgorithm):
	'''Baseline lossless compression
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.RAW = self.target + '/data'
		self.DATA_FILES += [self.RAW+ '.npyz']


	"""The main compression loop
	"""
	def compress(self):
		start = timer()


		np.save(self.RAW, self.data)

		compressz(self.RAW + '.npy', self.RAW+'.npyz')

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']

		

	def decompress(self, original=None):

		start = timer()

		decompressz(self.RAW + '.npyz', self.RAW+'.npy')
		codes = np.load(self.RAW + '.npy')

		normalization = np.load(self.NORMALIZATION + '.npy')
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])

		for i in range(p):
			codes[:,i] = (codes[:,i])# + normalization[1,i])#*(normalization[0,i] - normalization[1,i])


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes


class BitStripGZ(CompressionAlgorithm):
	'''Baseline lossy compression
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.RAW = self.target + '/data'
		self.DATA_FILES += [self.RAW+ '.npyz']


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		if self.error_thresh >= 1e-3:
			np.save(self.RAW, self.data.astype(np.float16))
		elif self.error_thresh >= 1e-6:
			np.save(self.RAW, self.data.astype(np.float32))
		else:
			np.save(self.RAW, self.data)

		compressz(self.RAW + '.npy', self.RAW+'.npyz')

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']

		

	def decompress(self, original=None):

		start = timer()

		decompressz(self.RAW + '.npyz', self.RAW+'.npy')
		codes = np.load(self.RAW + '.npy')

		normalization = np.load(self.NORMALIZATION + '.npy')
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])

		for i in range(p):
			codes[:,i] = (codes[:,i])# + normalization[1,i])#*(normalization[0,i] - normalization[1,i])


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes


	#zero out as many bits as possible
	def _strip_code(self, vector):
		p = vector.shape[0]

		for i in range(p): #go component by component
			value = vector[i]
			ba = bytearray(struct.pack("d", value))

			for j in range(len(ba)):
				tmp = ba[j]
				ba[j] = int('00000000')
				newvalue = struct.unpack("d", ba)[0]

				if np.abs(newvalue - value) > self.error_thresh:
					ba[j] = tmp
					vector[i] = struct.unpack("d", ba)[0]
					break

		return vector
		

####
"""
Test code here
"""
####

"""
data = np.loadtxt('/Users/sanjaykrishnan/Downloads/test_comp/ColorHistogram.asc')[:,1:]

#normalize this data
N,p = data.shape


nn = BitStripGZ('quantize')
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)
"""


