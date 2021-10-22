'''This module defines the integer-matrix subroutines used in
   quarc.
'''
from core import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np


class LinearAutoregressiveIM():

	def __init__(self):
		pass


	def compress(self, src):
		data = src.copy()
		N,p = data.shape
		coderange = np.max(src)
		models = []
		
		for j in range(1, p):
			slice = list(range(0,j))
			prediction = j


			X = src[:,slice]
			Y = src[:,prediction]

			value,counts = np.unique(src[:,prediction], return_counts=True)
			median_count = np.median(counts)
			lookup = {v:(c < median_count)*1.0 for v,c in zip(value,counts)}


			reg = LogisticRegression(class_weight=lookup)
			reg.fit(X,Y)
			Ypred = reg.predict(X)

			mask = (Ypred.reshape(-1) == Y.reshape(-1))

			data[mask, j] = coderange + 1

			#if it makes a difference
			#if entropy(data[:, j]) > entropy(src[:, j]):
			#	data[mask, j] = src[mask, j]
			#else:
			models.append(np.hstack([reg.intercept_.reshape(-1,1),reg.coef_]))

		return data, models



	def compress(self, src):
		data = src.copy()
		N,p = data.shape
		coderange = np.max(src)
		models = []
		classes = []
		
		for j in range(1, p):
			slice = list(range(0,j))
			prediction = j


			X = src[:,slice]
			Y = src[:,prediction]

			value,counts = np.unique(src[:,prediction], return_counts=True)
			median_count = np.median(counts)
			lookup = {v:(c < median_count)*1.0 for v,c in zip(value,counts)}


			reg = LogisticRegression(class_weight=lookup)
			reg.fit(X,Y)
			Ypred = reg.predict(X)

			mask = (Ypred.reshape(-1) == Y.reshape(-1))

			data[mask, j] = coderange + 1

			#is it worth it?
			#cost of saving parameters
			cost_estimate = j * 32 * len(reg.classes_) #bits
			savings = (entropy(src[:, j]) - entropy(data[:, j]))*N

			print('Model Cost Estimate (bits)', cost_estimate, 'Projected Savings (bits)', savings)

			#if it makes a difference
			if savings  <= cost_estimate:
				data[mask, j] = src[mask, j]
				models.append(None)
				classes.append(None)
			else:
				models.append(np.hstack([reg.intercept_.reshape(-1,1),reg.coef_]))
				classes.append(reg.classes_)

			#print(reg.classes_.shape, reg.coef_.shape, reg.intercept_.shape)

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
