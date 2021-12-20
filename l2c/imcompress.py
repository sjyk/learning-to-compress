'''This module defines the integer-matrix subroutines used in
   quarc.
'''
from core import *
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


class LinearAutoregressiveIM():

	def __init__(self):
		pass

	def avg_run_size(self, vector):

		N = vector.shape[0]

		runs = []
		start = 0
		for i in range(N-1):
			if vector[i] != vector[i+1]:
				runs.append(i+1 - start)
				start = i + 1

		return np.mean(runs)


	def compress(self, src):
		data = src.copy()
		N,p = data.shape
		coderange = np.max(src)

		models = []
		classes = []

		lag = 1
		
		for j in range(0, p):

			training_X = []
			training_Y = []

			for i in range(lag, N):
				training_X.append(src[i-lag:i,:].flatten())
				training_Y.append(src[i,j])

			X = np.array(training_X)
			Y = np.array(training_Y)

			p = PolynomialFeatures(degree = 2)
			X  = p.fit_transform(X)

			reg = LinearRegression()
			reg.fit(X,Y)

			Y_pred = np.round(reg.predict(X))

			cnt = 0
			for i in range(lag, N-1):
				if int(Y_pred[i-lag]) == int(Y[i-lag]):
					data[i,j] = coderange + 1
					cnt += 1

					#print(int(Y_pred[i-lag]),int(Y[i-lag]))

			print('Replaced', cnt, N, 'for attr',j, self.avg_run_size(src[:,j]),self.avg_run_size(data[:,j]))
			#models.append(reg.coef_)
			break



		return data, models, classes


	def decompress(self, src, models, classes):

		coderange = np.max(src)
		N,p = src.shape

		#left to right decoding
		for (j,(m,c)) in enumerate(zip(models, classes)):
			if not m is None:
				slice = list(range(0,j))
				X = src[:,slice]

				reg = LogisticRegression()
				reg.intercept_ = m[:,0]
				reg.coef_ = m[:,1:]
				#print(c, m.shape, c.shape)
				reg.classes_ = c

				Ypred = reg.predict(X)

				#Ypred = np.array([classes[Ypred[i]] for i in range(N)])#assign actual class values

				mask = (src[:,j] == coderange)
				src[mask,j] = Ypred[mask]#impute

		return src, np.max(src)




class LinearAutoregressiveIM2():

	def __init__(self):
		pass


	def compress(self, src):
		data = src.copy()
		N,p = data.shape
		coderange = int(np.max(src))
		models = []
		classes = []
		
		data = data.flatten()

		new_data = np.zeros((N*p, coderange))

		#for each possible code
		for i in range(coderange):
			new_data[data == i, i] = 1
			
		#	new_data[:,i] = data[]

		return new_data, models, classes


	def decompress(self, src, models, classes):

		coderange = np.max(src)
		N,p = src.shape

		#left to right decoding
		for (j,(m,c)) in enumerate(zip(models, classes)):
			if not m is None:
				slice = list(range(0,j))
				X = src[:,slice]

				reg = LogisticRegression()
				reg.intercept_ = m[:,0]
				reg.coef_ = m[:,1:]
				#print(c, m.shape, c.shape)
				reg.classes_ = c

				Ypred = reg.predict(X)

				#Ypred = np.array([classes[Ypred[i]] for i in range(N)])#assign actual class values

				mask = (src[:,j] == coderange)
				src[mask,j] = Ypred[mask]#impute

		return src, np.max(src)