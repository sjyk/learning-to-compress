import numpy as np
import os
from timeit import default_timer as timer
import math
from core import *
from imcompress import *
import glob
import pickle


#Quantized Autoregressive Codes (Quarc)
class Quarc(CompressionAlgorithm):
	'''Applies a basic quantization algorithm to compress the data
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005):

		super().__init__(target, error_thresh)

		self.coderange = int(math.ceil(1.0/error_thresh))

		self.AR_MODELS = target + '/learned'
		self.AR_CLASSES = target + '/classes'


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		codes = np.ones((self.N, self.p))*-1#set all to negative one

		for i in range(self.N):
			for j in range(self.p):
				codes[i,j] = int(self.data[i,j]*self.coderange)


		lin = LinearAutoregressiveIM()
		codes, models, classes = lin.compress(codes)
		for i, model in enumerate(models):

			if model is None:
				continue

			fname = self.AR_MODELS + '_' + str(i+1)
			np.save(fname, model)
			compressz(fname +'.npy', fname+'.gz')
			self.DATA_FILES += [fname+'.gz']

		struct = iarray_bitpacking(codes, order='F')
		struct.flushz(self.CODES)
		struct.flushmeta(self.METADATA)

		classes.insert(0,None)
		pickle.dump(classes, open(self.AR_CLASSES, 'wb'))

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']
		self.compression_stats['code_size'] = self.getSize() - self.getModelSize()
		self.compression_stats['model_size'] = self.getModelSize()
	

	def decompress(self, original=None):

		start = timer()

		struct = BitPackedStruct.loadmeta(self.METADATA)
		codes = struct.loadz(self.CODES)
		coderange = np.max(codes)

		normalization = np.load(self.NORMALIZATION + '.npy')
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])
		bit_length = struct.bit_length


		#pass the parameters back to the decoder
		lin = LinearAutoregressiveIM()
		models = [None]*p
		for f in glob.glob(self.AR_MODELS +'*'):
			fname = f.split('.')[0]
			decompressz(f, fname +'.npy')
			model_index = int(fname.split('_')[1])
			models[model_index] = np.load(fname +'.npy')

		classes = pickle.load(open(self.AR_CLASSES,'rb'))

		codes, coderange = lin.decompress(codes, models, classes)


		for i in range(p):
			codes[:,i] = (codes[:,i]/coderange + normalization[1,i])*(normalization[0,i] - normalization[1,i])


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)


		self.compression_stats.update(struct.additional_stats)

		return codes





####
"""
Test code here
"""
####

data = np.loadtxt('/Users/sanjaykrishnan/Downloads/test_comp/ColorHistogram.asc')[:,1:]

#normalize this data
N,p = data.shape


nn = Quarc('quantize')
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)


