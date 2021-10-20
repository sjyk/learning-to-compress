import numpy as np
import os
from timeit import default_timer as timer
import math
import struct
from sklearn.ensemble import RandomForestRegressor #equivalent to CaRT
from core import *
import pickle


class Spartan(CompressionAlgorithm):
	'''Spartan Algorithm
	'''


	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005, prediction_fraction=0.75):

		super().__init__(target, error_thresh)

		self.RAW = self.target + '/data'
		self.MODEL =  self.target + '/learned'
		self.DATA_FILES += [self.RAW + '.npyz', self.MODEL +'.gz']
		self.prediction_fraction = prediction_fraction


	"""The main compression loop
	"""
	def compress(self):
		start = timer()

		src = self.data.copy()

		N,p = self.data.shape
		indexes = [(entropy(self.data[:,i]), i) for i in range(p)]
		indexes.sort()
		indexes = [ i for _,i in indexes]

		model_slice = indexes[0:int(p*self.prediction_fraction)]
		prediction_slice = indexes[int(p*self.prediction_fraction):]

		models = []

		reg = RandomForestRegressor(n_estimators=1, max_depth=10)
		reg.fit(src[:, model_slice], src[:,prediction_slice])
		preds = reg.predict(src[:, model_slice])

		models.append(reg)

		for j, prediction_index in enumerate(prediction_slice):
			
			for i in range(N):
				if np.abs(preds[i,j]-src[i, prediction_index]) < self.error_thresh:
					self.data[i,prediction_index] = -1 #set it equal to negative 1

			
		np.save(self.RAW, self.data)
		compressz(self.RAW + '.npy', self.RAW+'.npyz')

		pickle.dump((models,indexes), open(self.MODEL,'wb'))

		compressz(self.MODEL, self.MODEL+'.gz')

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']
		self.compression_stats['code_size'] = self.getSize() - self.getModelSize()
		

	def decompress(self, original=None):

		start = timer()

		decompressz(self.RAW + '.npyz', self.RAW+'.npy')
		codes = np.load(self.RAW + '.npy')

		decompressz(self.MODEL+'.gz', self.MODEL)
		models, indexes = pickle.load(open(self.MODEL,'rb'))

		normalization = np.load(self.NORMALIZATION + '.npy')
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])

		model_slice = indexes[0:int(p*self.prediction_fraction)]
		prediction_slice = indexes[int(p*self.prediction_fraction):]
		preds = models[0].predict(codes[:,model_slice])

		for j, prediction_index in enumerate(prediction_slice):
			
			for i in range(N):
				if codes[i, prediction_index] < 0:
					codes[i, prediction_index] = preds[i,j]


		for i in range(p):
			codes[:,i] = (codes[:,i] + normalization[1,i])*(normalization[0,i] - normalization[1,i])


		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes



####
"""
Test code here
"""
####

data = np.loadtxt('/Users/sanjaykrishnan/Downloads/test_comp/ColorHistogram.asc')[:,1:]

#normalize this data
N,p = data.shape


nn = Spartan('quantize')
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)


