
'''
Author: David Lepore, Alex Hutman, Stephen Kern
Date: Spring 2019
Course: CSC350 (Intelligent Systems)

This is our digit classifier program. When its defined it needs to be given the k groups of the data set, 
and the path of the data in order to read it in. 
Using this data it is put through a model passed from TestProject.py through fitting training data, and predicting outputs for the testing data.
The training and testing process has 10 folds, meaning that with each run, there are 9 groups of training and one group of testing. 
This process is repeated 10 times for each group in order for each group to be apart of testing and training once . 
eg: 0 is testing and 1-9 is training 
	1 is testing and 0, 2-9 is training ... etc
'''

import sys
import re
from os import listdir
from os.path import isfile, join
import json
from sklearn import metrics as jeff
import math
import numpy as np
from sklearn import svm
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn.neighbors
from joblib import dump, load
import csv
from sklearn.metrics import matthews_corrcoef

""" Convert a digit to an array of 0's and a 1.  E.g. 3 converts to 0, 0, 0, 1, 0, 0, 0, ...."""
def convert(digit, num_digits):
	d = [0]*num_digits
	d[digit] = 1
	return np.array(d)

class DigitClassifiers:
	def __init__(self, data, base_path, model_type, isCompetition):
		self.data = data
		all_digits_path = base_path + "all_digits/"
		model_path = base_path + "models/test_model"
		challenge_path = base_path + "FinalChallengeSetJSON/"

		if not isCompetition: 
			num_digits = 10
			sum_matrices = None
			layers = [1024,395,10] #We looped through [1024,x,10] for all x in [10...1024] and 395 was the best... but is still bad. We tried.
			#for each k group there is a fold 
			for i in range(len(self.data)):
				#poping the i group for testing, and the rest for training
				x_test = self.data.pop(i)
				x_train = self.data
				#Getting all data from each group in x_train
				x_train_data = [[self.read_data(all_digits_path + i) for i in j] for j in x_train]
				#Finds all matches of the regular expression in x_train, and then grabs all data from the first group of the expression into y_train_data
				# y_train data holds all middle digits of the file names which tells if that file is a 1 or 0
				y_train_arr = [[re.search("^input_[0-9]+_([0-9]+)_[0-9]+\.json$", i) for i in x_train[j]] for j in range(len(x_train))]
				y_train_data = [[int(i.group(1)) for i in y_train_arr[j] if i] for j in range(len(y_train_arr))]
				#reshape x_train data and y_train data into 2d and single dimensions arrays accordingly
				x_train_data = np.array(x_train_data).reshape((-1,1024))
				y_train_data = np.array(y_train_data).reshape((-1))

				#creates model, and then feeds data above to fit the model, with 10 epochs and a batch size of 8 if its a neural network model
				model = self.create_model(1024,layers, 'sigmoid', model_type)

				if model_type == 'neural':
					y_train_data_adj = np.array([convert(digit, num_digits) for digit in y_train_data])
					model.fit(x_train_data, y_train_data_adj, epochs=10, batch_size=6)
				else:
					model.fit(x_train_data, y_train_data)
				
				#repeats the same process above expect of the one training group
				x_test_data = [self.read_data(all_digits_path + i) for i in x_test]   # Random input data
				y_test_arr = [re.search("^input_[0-9]+_([0-9]+)_[0-9]+\.json$", i) for i in x_test]
				y_test_data = [int(i.group(1)) for i in y_test_arr if i]
				#reshapes the training group
				x_test_data = np.array(x_test_data).reshape((-1,1024))
				y_test_data = np.array(y_test_data).reshape((-1))

				# Evaluate the model from a sample test data set
				y_predict = model.predict(x_test_data)

				if model_type == 'neural':
					y_predict = np.array([np.argmax(p) for p in y_predict])
				#build the confusion matrix and find the total MCC score
				confusion_matrix = jeff.confusion_matrix(y_test_data, y_predict)
				totalMcc = matthews_corrcoef(y_test_data, y_predict)  
				sum_matrices = np.zeros(confusion_matrix.shape) if i==0 else np.add(sum_matrices, confusion_matrix)
				#adds the testing group back into the data set 
				self.data.insert(len(self.data),x_test)
			"""
			This portion gets the True postives, negatives, as well as False postives, and negatives  for each digit.
			It then uses this to find the Precision, recall, f1 and MCC scores for each digit
			Prints all results to an "model_name".CSV files
			"""
			tp = [sum_matrices[i][i] for i in range(0, len(sum_matrices))]
			fp = [0]*len(sum_matrices)
			fn = [0]*len(sum_matrices)
			for d in range(0, len(sum_matrices)):
				for d2 in range(0, len(sum_matrices)):
					if d != d2:
						fp[d] += sum_matrices[d2][d]
						fn[d] += sum_matrices[d][d2]
			sum = np.sum(sum_matrices)
			tn = [sum - tp[i] - fp[i] - fn[i] for i in range(0, len(sum_matrices))]
			mcc = [((tp[i]*tn[i])-fp[i]*fn[i])/math.sqrt((tp[i]+fp[i])*(tp[i]+fn[i])*(tn[i]+fp[i])*(tn[i]+fn[i])) for i in range(0, len(sum_matrices))]

			precision = [tp[i]/(tp[i] + fp[i]) for i in range(len(tp))]
			recall = [tp[i]/(tp[i] + fn[i]) for i in range(len(tp))]
			f1 = [2*precision[i]*recall[i]/(precision[i]+recall[i]) for i in range(len(tp))]
			with open(model_type+'.csv', mode='w') as file:
				writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				writer.writerow(["Digit", "Precision", "Recall", "F1", "MCC"])
				for i in range(len(sum_matrices)):
					writer.writerow([i, precision[i], recall[i], f1[i], mcc[i]])

				writer.writerow([sum_matrices[i] for i in range(len(sum_matrices))])
				writer.writerow([totalMcc])
			dump(model, model_path)
		#This is for the competition, simply does training on digit data set and then tests using finalChallengeJson dataset with no k cross validation
		else:
			x_train = self.data
			x_train_data = [self.read_data(all_digits_path + i) for i in self.data]

			x_train_data = np.array(x_train_data).reshape((-1,1024))


			y_train_arr = [re.search("^input_[0-9]+_([0-9]+)_[0-9]+\.json$", i) for i in x_train]
			y_train_data = [int(i.group(1)) for i in y_train_arr if i]
			y_train_data = np.array(y_train_data).reshape((-1))

			model = self.create_model(1024,[512,512,10], 'sigmoid', model_type)
			model.fit(x_train_data, y_train_data)

			files_regexp = [re.search("^input([0-9]+)\.json$", f) for f in listdir(challenge_path) if isfile(join(challenge_path, f))]
			files = [i.group(0) for i in files_regexp if i]

			numbers = [int(i.group(1)) for i in files_regexp if i]
			data = [self.read_data(challenge_path + i) for i in files]
			data = np.array(data).reshape((-1,1024))
			predict = model.predict(data)


			with open(model_type+'.csv', mode='w') as file:
				writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				for i in range(len(numbers)):
					writer.writerow([numbers[i],predict[i]])

	""" 
	Reads in the given JSON file as outlined in the README.txt file.
	"""

	def read_data(self, file):
		try:
			with open(file, 'r') as inf:
				bitmap = json.load(inf)

			return bitmap
		except FileNotFoundError as err:
			print("File Not Found: {0}.".format(err))
	#creates model based on passed parameters
	def create_model(self, in_dim, units, activation, model_type):
		assert model_type in ['neural','svm','bayes', 'kNN','LDA'] #If not one of these, throw an error
		
		if model_type == 'neural':
			model = Sequential()
			model.add(Dense(units=units[0], input_dim=in_dim, kernel_initializer='normal', activation=activation))
			for i in units[1:]:
				model.add(Dense(units=i, kernel_initializer='normal', activation=activation))
			model.compile(loss='mean_squared_error',
		    	          optimizer='sgd',
		    	          metrics=['accuracy'])
		elif model_type == 'svm':
			model = svm.LinearSVC()
		elif model_type == 'bayes':
			model = GaussianNB()
		elif model_type == 'kNN':
			model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='kd_tree', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
		elif model_type == 'LDA':
			model = LinearDiscriminantAnalysis(solver="eigen", store_covariance=True)
		
		return model
