import numpy as np
import os
from timeit import default_timer as timer
import math
import struct
from core import *


class Gorilla(CompressionAlgorithm):
	'''Applies a rolling xor operation like Gorilla
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

		codes = self.data.copy() #store a copy of the dataset

		for i in range(self.N):
			self._strip_code(codes[i,:]) #zero out as many as possible	


		placeholder = codes.copy()

		for j in range(self.p):
			for i in range(1, self.N):
				codes[i,j] = self._float_xor(placeholder[i-1,j],codes[i,j])


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

		
		for i in range(1, self.N):
			for j in range(p):
				codes[i,j] = self._float_xor(codes[i-1,j],codes[i,j])
				print(codes[i,j])
		
		for j in range(p):
			codes[:,j] = (codes[:,j])*(normalization[0,j] - normalization[1,j]) + normalization[1,j]


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes


	#zero out as many bits as possible to hit error threshold
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


		#zero out as many bits as possible to hit error threshold
	def _float_xor(self, v1, v2):
		ba1 = bytearray(struct.pack("d", v1))
		ba2 = bytearray(struct.pack("d", v2))

		for j in range(len(ba1)):
			ba2[j] = ba1[j] ^ ba2[j]

		return struct.unpack("d", ba2)[0]



####
"""
Test code here
"""
####

"""
data = np.loadtxt('/Users/sanjaykrishnan/Downloads/HT_Sensor_UCIsubmission/HT_Sensor_dataset.dat')[:2000,1:]
#data = np.load('/Users/sanjaykrishnan/Downloads/ts_compression/l2c/data/electricity.npy')
print(data.shape)
#data = np.nan_to_num(data)

#normalize this data
N,p = data.shape


nn = Gorilla('quantize')
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)
"""


