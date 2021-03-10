
'''
Author: David Lepore, Alex Hutman, Stephen Kern
Date: Spring 2019
Course: CSC350 (Intelligent Systems)

This is our digit classifier program. When its defined it needs to be given the k groups of the data set, 
and the path of the data in order to read it in. 
Using this data it is put through a neural network model through fitting training data, and predicting outputs for the testing data.
The training and testing process has 10 folds, meaning that with each run, there are 9 groups of training and one group of testing. 
This process is repeated 10 times for each group in order for each group to be apart of testing and training once . 
eg: 0 is testing and 1-9 is training 
	1 is testing and 0, 2-9 is training ... etc
'''



from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import sys
import re
import json
from sklearn import metrics as jeff
import math
import csv

class DigitClassifier:

	def __init__(self, data, path, model_type):
		self.data = data
		# totals that add up each square in the confusion matrix, along with the max score
		totalTruePositives = 0;
		totalFalseNegatives= 0;
		totalFalsePostives = 0;
		totalTrueNegatives = 0;
		maxScore = 0;
		confusion_matrices = []
		#for each k group there is a fold 
		for i in range(len(self.data)):
			#poping the i group for testing, and the rest for training
			x_test = self.data.pop(i)
			x_train = self.data
		
			#Getting all data from each group in x_train
			x_train_data = [[self.read_data(path + i) for i in j] for j in x_train]
			#Finds all matches of the regular expression in x_train, and then grabs all data from the first group of the expression into y_train_data
			# y_train data holds all middle digits of the file names which tells if that file is a 1 or 0
			y_train_arr = [[re.search("^input_[0-9]+_([0-9]+)_[0-9]+\.json$", i) for i in x_train[j]] for j in range(len(x_train))]
			y_train_data = [[int(i.group(1)) for i in y_train_arr[j] if i] for j in range(len(y_train_arr))]
			
			#reshape x_train data and y_train data into 2d and single dimensions arrays accordingly
			x_train_data = np.array(x_train_data).reshape((-1,1024))
			y_train_data = np.array(y_train_data).reshape((-1))

			#creates model, and then feeds data above to fit the model, with 10 epochs and a batch size of 8
			model = self.create_model(1024,[1024,50,1], 'sigmoid', model_type)
			model.fit(x_train_data, y_train_data, epochs=10, batch_size=8)
		

			#repeats the same process above expect of the one training group
			x_test_data = [self.read_data(path + i) for i in x_test]  
			y_test_arr = [re.search("^input_[0-9]+_([0-9]+)_[0-9]+\.json$", i) for i in x_test]
			y_test_data = [int(i.group(1)) for i in y_test_arr if i]
			#reshapes the training group
			x_test_data = np.array(x_test_data).reshape((-1,1024))
			y_test_data = np.array(y_test_data).reshape((-1))


		
			# Evaluate the model from a sample test data set, and makes a prediction for each 
			y_predict = model.predict(x_test_data)
			#if the prediction score is closer to 0 it guesses 0, if its closer to 1 then it guesses 1
			y_predict = np.array([ round(p[0]) for p in y_predict])

			#creates a confusion matrix from the test data and the predictions, and accumulates each score in each index of the confusion matrix 
			confusion_matrix = jeff.confusion_matrix(y_test_data, y_predict)
			confusion_matrices.append(confusion_matrix)
			tp = confusion_matrix[0][0]
			fp = confusion_matrix[0][1]
			fn = confusion_matrix[1][0]
			tn = confusion_matrix[1][1]
			totalTruePositives +=tp
			totalFalsePostives +=fp
			totalFalseNegatives += fn
			totalTrueNegatives += tn
			#adds the testing group back into the data set 
			self.data.insert(len(self.data),x_test)
		
		#Calculates the mccscore based on the total aggergate of the confusion matrix 
		print(totalTrueNegatives, totalTruePositives, totalFalsePostives, totalFalseNegatives)
		totalmcc = ((totalTruePositives*totalTrueNegatives)-totalFalsePostives*totalFalseNegatives)/math.sqrt((totalTruePositives+totalFalsePostives)*(totalTruePositives+totalFalseNegatives)*(totalTrueNegatives+totalFalsePostives)*(totalTrueNegatives+totalFalseNegatives))
		print(totalmcc)
		with open(model_type+'.csv', mode='w') as file:
				writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				for i in range(len(confusion_matrices)):
					writer.writerow([confusion_matrices[i],i])
				writer.writerow([totalmcc])


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

	def create_model(self, in_dim, units, activation, model_type):
		assert model_type in ['neural','svm','bayes', 'kNN','LDA']
		
		if model_type == 'neural':
			model = Sequential()
			model.add(Dense(units=units[0], input_dim=in_dim))
			model.add(Activation(activation))
			#For every unit given besides the first one it sets the number of neurons for that hidden layer along with its activation function
			for i in units[1:]:
				model.add(Dense(units=i)) 
				model.add(Activation(activation))
			model.compile(loss='mean_squared_error',
		    	          optimizer='sgd',
		    	          metrics=['accuracy'])
		
		return model
"""
Creates model.
We use a sequential model with three hidden layers. 
The input dimensions would be 1024, for each row and column within a JSon file
First hidden layer holds: 1024 neurons,
Second hidden layer holds: 50 neurons,
Last hidden layer hodls: a single output neuron 
We use the sigmoid activation function 
We compile the model from 'mean_squared_error for loss, sgd for optimizer, and accuracy for metrics'
"""