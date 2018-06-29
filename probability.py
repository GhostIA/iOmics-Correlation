import pandas as pd
import numpy as np
from MLP import MLP
import sklearn as sklearn
import scipy.stats.mstats as sp
z = 5
activation = np.tanh(np.log(1 + np.exp(z)))
generator = MLP(num_inputs = 436, num_outputs = 436, num_hl=75,  num_hlnodes=50, af = activation)
discriminator = MLP(num_inputs = 436 , num_outputs = 1, num_hl=75,  num_hlnodes=50, af = activation)


def train(epochs = 200, *args, K = 1):
	def mean_vectors(*args):
		training_list = []
		vector_sums = []
		length_of_vectors = []
		average_m = [] 
		for train_set in args:
			training_list.append(train_set) #training sets in one
		for item in training_list:
			sum = 0
			length_of_vectors.append(len(item))
			for i in range(0, len(item)):
				sum = sum + item[i]
				vector_sums.append(sum)
		for i in range(0, len(length_of_vectors)):
			average = vector_sums[i]/length_of_vectors[i]
			average_m.append(average)
		return average_m
	m = mean_vectors(*args)
	cov_matrix = np.cov(*args)
	noise_prior = np.random.normal(m, cov_matrix)
	num_training_sets = len(m)
	for i in range(0, len(epochs) - 1):
		momentum_vectors_g = []
		momentum_vectors_d = []
		for i in range(0, K):
			delta = []
			J = 0


