import numpy as np
from six.moves import cPickle as pickle
from six.moves import range
from parameters import *
import sys
from math import sqrt

from sklearn.metrics import accuracy_score

pickle_file = 'lesion.pickle'

with open(pickle_file, 'rb') as f:
	save = pickle.load(f)
	features = save['features']
	labels = save['labels']

import matplotlib.pyplot as plt
import random
# Will take 80% of patients randomly as training data
# Use 18 patients as training data, 5 as testing data
training_indices = []
while len(training_indices) != 18:
	index = random.randint(0, 22)
	if index not in training_indices:
		training_indices.append(index);
print training_indices

testing_indices = []
for i in range(0, 23):
	if i not in training_indices:
		testing_indices.append(i);
print testing_indices

# Preprocess data for training data of 18 patients
ftrain = np.array([])
ltrain = np.array([])
for i in training_indices:
	for j in range(0, len(features[i])):
		test1 = features[i][j]
		test1_flat = test1.ravel()
		test2 = labels[i][j]
		test2_flat = test2.ravel()
		ftrain = np.append(ftrain, test1)
		ltrain = np.append(ltrain, test2)
ftrain = ftrain.reshape(-1, 1)

# Preprocess data for testing data of 5 patients
ftrain2 = np.array([])
ltrain2 = np.array([])
for i in testing_indices:
	for j in range(0, len(features[i])):
		test1 = features[i][j]
		test1_flat = test1.ravel()
		test2 = labels[i][j]
		test2_flat = test2.ravel()
		ftrain2 = np.append(ftrain2, test1)
		ltrain2 = np.append(ltrain2, test2)
ftrain2 = ftrain2.reshape(-1, 1)

print "Done with processing data" 

from sklearn.svm import LinearSVC

clf = LinearSVC()

print("Fitting data")
clf.fit(ftrain, ltrain)
print("Predicting data")
pred = clf.predict(ftrain2)

print accuracy_score(pred, ltrain2)