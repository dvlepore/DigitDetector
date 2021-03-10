
'''
Author: Christian Duncan
Modification: David Lepore, Alex Hutman, Stephen Kern
Date: Spring 2019
Course: CSC350 (Intelligent Systems)

This sample code shows how one can read in our JSON image files in Python3.  It reads in the file and then outputs the two-dimensional array.
It applies a simple threshold test - if value is > threshold then it outputs a '.' otherwise it outputs an 'X'.  More refinement is probably better.
But this is for displaying images not processing or recognizing images.
'''

from os import listdir
from os.path import isfile, join
import re
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import sys
import random
from DigitClassifier import DigitClassifier
""" 
Prints out the given "image" (a 2-dimensional array).
This just replaces any values greater than a threshold with . and otherwise with an X.
"""
def print_image(img, threshold):
	for row in img:
		for pixel in row:
			print('.' if pixel > threshold else 'X', end='')
		print()  # Newline at end of the row
"""
K cross validation method: Splits data into k groups where each group holds a random set of files from the data set.
The k value used is k = 10. 
First it divides the length of data by k which represents the number of files per group
then assign data to a temp data.
then initialize an empty array of groups.
Then for k-1 times to obtain all training groups:
	train = a random sample number of files from temp data equal to numofFilesPerGroup
	temp_data = the remainder of what files were taken from temp data
	append train into groups array
lastly finish and append the last part of temp data remaining as the last group.
"""
def kCross(k, data):
	numofFilesPerGroup = round(len(data)/k)
	temp_data = data
	groups = []
	for i in range(k-1):
		train = random.sample(temp_data,numofFilesPerGroup)
		temp_data = list(set(temp_data)-set(train))
		groups.append(train)
	groups.append(temp_data)
	return groups

""" 
Main entry point.  Assumes all the arguments passed to it are file names.
For each argument, reads in the file and the prints it out.
First we take all files from the path holding all data.
then we find all matches to our regular expression into our matches list.
then we get all jsonfiles from that matches list.
Once this happens we create our random partitions using kCross
it returns 10 partitions from all json files 
Finally, we call digitclassifier using the data above, structuring into a neural network model
"""
def main():
	my01path = "C:/DigitProject/DigitDetector/binaryFolder/digit_data/"
	files = [f for f in listdir(my01path) if isfile(join(my01path, f))]
	matches = [re.search("^input_[0-9]+_[0-9]+_[0-9]+\.json$", i) for i in files]
	jsonFiles = [i.group(0) for i in matches if i]
	partitions = kCross(10, jsonFiles)
	DigitClassifier(partitions, my01path, 'neural')



# This is just used to trigger the Python to run the main() method.  The if statement is used so that if this code
# were imported as a module then everything would load but main() would not be executed.
if __name__ == "__main__":
	main()


