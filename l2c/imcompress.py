'''This module defines the integer-matrix subroutines used in
   quarc.
'''
from core import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class LinearAutoregressiveIM():

	def __init__(self):
		pass


	def compress(self, src):
		data = src.copy()
		N,p = data.shape
		coderange = np.max(src)

		indexes = [(entropy(src[]), i) for i in range(p)]
		indexes.sort()
		indexes = [i for _,i in indexes]

		#reorganize columns
		data = data[:, indexes]
		models = []


		value, counts = np.unique(data.flatten(), return_counts=True)
		weights = {int(v):1.0/c for v,c in zip(value,counts)}

		print('Before Entropy',entropy(data))

		for i in range(N):
			print(src[i,:])
		
		print(np.mean(src, axis=0))
		exit()
		

		for j in range(1, p):
			model_slice = indexes[0:j] #f(x_i | x_0,...,x_i-1)
			prediction = [indexes[j]]

			total = 0

			reg = RandomForestRegressor()
			X = src[:,model_slice]

			#poly = PolynomialFeatures(2)
			#X = poly.fit_transform(X)

			Y = src[:,prediction]
			W = [weights[int(y)] for y in Y]

			reg.fit(X, Y, W) #weighting
			preds = reg.predict(X)

			#print('A',j,entropy(np.round(Y - preds)))

			#print('B',j,entropy(Y))			

			#exit()


			for i in range(N):
				#print(src[i,:])
				pred_int = int(np.round(preds[i]))

				#if j > 2:
				#	print(pred_int, int(np.round(src[i,prediction])))

				if pred_int == int(np.round(src[i,prediction])) \
					and int(np.round(src[i,prediction])) != 0:
					data[i,j] =  coderange + 1 #highest value is sentinel
					#total +=1

				#print(src[i,:] - data[i,:])

			models.append(reg)

			#print(total/N)
			#if total/N < 0.1:
			#	data[:,prediction] = src[:,prediction]

		print(np.sum(data == coderange + 1)/(N*p))
		print(np.unique(data.flatten(), return_counts=True))
		print('After Entropy',entropy(data))

		return data, models
