import numpy as np
import os
from timeit import default_timer as timer
import math
from core import *
import struct


class ItCompress(CompressionAlgorithm):
	'''Applies the itcompress algorithm to the data
	'''

	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.0001):

		super().__init__(target, error_thresh)

		self.MODEL = self.target + '/learned'
		self.DATA_FILES += [self.target + '/model' + '.npyz']


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		codes = np.ones((self.N, 1))*-1#set all to negative one
		code_lst = []
		code_set = set()

		for i in range(self.N):
			#print(i)

			found_code = False

			for j in code_set:

				if np.max(np.abs(self.data[i,:]-self.data[j,:])) < self.error_thresh:
					codes[i] = j
					found_code = True
					break


			if not found_code:
				codes[i] = len(code_set)	
				code_lst.append(self._strip_code(self.data[i,:]))
				code_set.add(i)

		struct = iarray_bitpacking(codes)
		struct.flushz(self.CODES)
		struct.flushmeta(self.METADATA)

		model = np.vstack(code_lst)
		np.save(self.MODEL, model)
		compressz(self.MODEL + '.npy', self.MODEL+'.npyz')

		self.DATA_FILES += [self.MODEL+'.npyz']

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']
		self.compression_stats['code_size'] = self.getSize() - self.getModelSize()
		self.compression_stats['model_size'] = self.getModelSize()

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
		

	def decompress(self, original=None):

		start = timer()

		struct = BitPackedStruct.loadmeta(self.METADATA)
		codes = struct.loadz(self.CODES)

		decompressz(self.MODEL + '.npyz', self.MODEL+'.npy')
		model = np.load(self.MODEL + '.npy')

		normalization = np.load(self.NORMALIZATION + '.npy')
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])
		bit_length = struct.bit_length

		decoding_array = np.zeros((N,p))
		for i in range(N): #modified because of error
			decoding_array[i, :] = model[int(codes[i]),:] 


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, decoding_array)

		return codes


####
"""
Test code here
"""
####
# '/Users/brunobarbarioli/Documents/Research/learning-to-compress-master/l2c/data/electricity_nips/electricity.npy'
# '/Users/brunobarbarioli/Documents/Research/learning-to-compress-master/l2c/data/exchange_rate_nips/exchange_rate.npy'
# '/Users/brunobarbarioli/Documents/Research/learning-to-compress-master/l2c/data/solar_nips/solar.npy'
# '/Users/brunobarbarioli/Documents/Research/learning-to-compress-master/l2c/data/taxi_30min/taxi.npy'
# '/Users/brunobarbarioli/Documents/Research/learning-to-compress-master/l2c/data/traffic_nips/traffic.npy'
#data = np.load('/Users/brunobarbarioli/Documents/Research/learning-to-compress-master/l2c/data/wiki-rolling_nips/wiki.npy')

#normalize this data
"""
N,p = data.shape


nn = ItCompress('quantize')
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)
"""


