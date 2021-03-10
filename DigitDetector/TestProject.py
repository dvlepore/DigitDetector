
'''
Author: Christian Duncan
Edited By: Alex Hutman, David Lepore, Stephen Kern
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
from DigitClassifiers import DigitClassifiers



""" 
Prints out the given "image" (a 2-dimensional array).
This just replaces any values greater than a threshold with . and otherwise with an X.
"""
def print_image(img, threshold):
	for row in img:
		for pixel in row:
			print('.' if pixel > threshold else 'X', end='')
		print()  # Newline at end of the row

def kCross(k, data):
	num = round(len(data)/k)
	temp_data = data
	groups = []
	for i in range(k-1):
		train = random.sample(temp_data,num)
		temp_data = list(set(temp_data)-set(train))
		groups.append(train)
	groups.append(temp_data)
	return groups

""" 
Main entry point.  Assumes all the arguments passed to it are file names.
For each argument, reads in the file and the prints it out.
"""
def main():
	base_path = "C:/DigitProject/DigitDetector/"
	competition = False

	mypath = base_path + "all_digits/"
	files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	x = [re.search("^input_[0-9]+_[0-9]+_[0-9]+\.json$", i) for i in files]
	y = [i.group(0) for i in x if i]

	if not competition:
		partitions = kCross(10, y)
		DigitClassifiers(partitions, base_path, 'LDA', competition)
	else:
		DigitClassifiers(y, base_path, 'svm', competition)


  

# This is just used to trigger the Python to run the main() method.  The if statement is used so that if this code
# were imported as a module then everything would load but main() would not be executed.
if __name__ == "__main__":
	main()


