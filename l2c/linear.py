import numpy as np
import os
from timeit import default_timer as timer
import math
import struct
from core import *


#implements a univariate sketch
class HierarchicalSketch():

	def __init__(self, min_error_thresh, blocksize):
		self.error_thresh = min_error_thresh
		self.blocksize = blocksize #must be a power of 2
		self.d = int(np.log2(blocksize))
		self.decode_matrices = {}

		for i in range(0,self.d+1):
			self.decode_matrices[i] = self.decode_matrix(self.blocksize,2**i)


	#allows for vectorized decoding
	def decode_matrix(self, n, d):
		H = np.zeros((n,d))
		width = (n // d)

		for j in range(d):
			H[j*width:(j+1)*width,j] = 1
			
		return H


	#maps a function over a window and calculates the max residual
	def _mapwindow(self, data, w, fn):
		width = (self.blocksize // w)
		W = np.zeros(w)
		R = np.zeros(w)
		for i in range(w):
			slc = data[i*width:(i+1)*width]
			W[i] = fn(slc)
			R[i] = np.max(np.abs(slc - W[i]))

		#print(W,R)
		return W, R


	#only works on univariate data
	def encode(self, data, fn=np.mean):

		curr = data.copy()
		hierarchy = [] 
		residuals = []

		for i in range(0, self.d + 1):
			v, r = self._mapwindow(curr, 2**i, fn) #map the window
			Hp = self.decode_matrices[i]
			curr -= np.dot(Hp, v) #fix with tile

			#zero out all buckets where max residual is less than eerror thresh
			
			mask = np.dot(Hp, (r < self.error_thresh)).astype(np.bool)
			curr[mask] = 0

			if i == self.d: #on the last level cut all small changes
				v[np.abs(v) <= self.error_thresh] = 0

			hierarchy.append(v)
			residuals.append(np.max(r))


		return list(zip(hierarchy, residuals))


	def decode(self, sketch, error_thresh=0):
		W = np.zeros(self.blocksize)
		for h,r in sketch:
			dims = h.shape[0]

			#print(h,r)

			index = int(np.log2(dims))
			Hp = self.decode_matrices[index]
			W += np.dot(Hp, h)

			#W += np.tile(h, self.blocksize // dims)

			if r < error_thresh:
				break

		return W

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


	#packs all of the data into a single array
	def pack(self, sketch):
		vectors = []
		for h,r in sketch:
			vector = np.concatenate([np.array([r]), h])
			vectors.append(vector)
		return np.concatenate(vectors)

	#unpack all of the data
	def unpack(self, array, error_thresh=0):
		array = array.copy()
		sketch = []
		for i in range(self.d+1):
			r = array[0]
			h = array[1:2**i+1]
			sketch.append((h,r))
			array = array[2**i+1:]

			if r < error_thresh:
				break

		return sketch


class MultivariateHierarchical(CompressionAlgorithm):

	'''
	The compression codec is initialized with a per
	attribute error threshold.
	'''
	def __init__(self, target, error_thresh=0.005, blocksize=1024):

		super().__init__(target, error_thresh)
		self.blocksize = blocksize
		self.sketch = HierarchicalSketch(self.error_thresh, blocksize)


	def compress(self):
		start = timer()

		arrays = []
		for j in range(self.p):
			vector = self.data[:,j].reshape(-1)
			en = self.sketch.encode(vector, fn=np.median)
			#print(en)
		
			arrays.append(self.sketch.pack(en))
	
		codes = np.vstack(arrays)
		#print('sizes', codes.shape)
		fname = self.CODES
		
		#np.save(fname, codes)
		np.savez_compressed(fname, a=codes)
		#compressz(self.CODES + '.npy', self.CODES+'.npyz')
		self.DATA_FILES += [self.CODES + '.npz']

		self.compression_stats['compression_latency'] = timer() - start
		self.compression_stats['compressed_size'] = self.getSize()
		self.compression_stats['compressed_ratio'] = self.getSize()/self.compression_stats['original_size']
		#self.compression_stats.update(struct.additional_stats)


	def decompress(self, original=None, error_thresh=0):

		start = timer()

		normalization = np.load(self.NORMALIZATION + '.npy')
		_, P2 = normalization.shape

		p = int(P2 - 1)
		N = int(normalization[0,p])
		codes = np.zeros((N,p))

		#decompressz(self.CODES + '.npyz', self.CODES+'.npy')
		packed = np.load(self.CODES+".npz", allow_pickle=False)['a']
		#packed = packed.reshape(-1, p)
		

		print(timer() - start)

		for j in range(self.p):
			#print('a',j,timer() - start)
			sk = self.sketch.unpack(packed[j,:], error_thresh)
			#print('b',j,timer() - start)
			codes[:,j] = self.sketch.decode(sk, error_thresh) 
			#print('c',j, timer() - start)


		for i in range(p):
			codes[:,i] = (codes[:,i])*(normalization[0,i] - normalization[1,i]) + normalization[1,i]

		print(timer() - start)

		self.compression_stats['decompression_latency'] = timer() - start

		if not original is None:
			#print(original-codes)
			self.compression_stats['errors'] = self.verify(original, codes)

		return codes

####
"""
Test code here
"""
####

data = np.loadtxt('/Users/sanjaykrishnan/Downloads/HT_Sensor_UCIsubmission/HT_Sensor_dataset.dat')[:1024,1:]

#data = np.load('/Users/sanjaykrishnan/Downloads/ts_compression/l2c/data/electricity.npy')
print(data.shape)
#data = np.nan_to_num(data)

#normalize this data
N,p = data.shape


nn = MultivariateHierarchical('hier')
nn.load(data)
nn.compress()
nn.decompress(data)
print(nn.compression_stats)




"""
data = np.loadtxt('/Users/sanjaykrishnan/Downloads/HT_Sensor_UCIsubmission/HT_Sensor_dataset.dat')[:1024,1:2]
#data = np.load('/Users/sanjaykrishnan/Downloads/ts_compression/l2c/data/electricity.npy')
print(data.shape)
#data = np.nan_to_num(data)

#normalize this data
N,p = data.shape


#l = np.arange(0,32).astype(np.float64)
h = HierarchicalSketch(0.00162203, 1024)
data = data.reshape(-1)
en = h.encode(data, fn=np.median)
decode = h.decode(en)
print(np.max(np.abs(decode-data)))

codes = h.pack(en) 

fname='comp'
np.save(fname, codes)
command = " ".join([nn.TURBO_CODE_LOCATION, nn.TURBO_CODE_PARAMETER, fname+".npy", fname+".npy"])
os.system(command)

#print(unsketch(sk, 1024), data)
#print(np.max(np.abs(unsketch(sk, 1024)-data))/(u-l))
#print((u-l)*0.005)
"""



