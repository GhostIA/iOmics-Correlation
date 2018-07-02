import pandas as pd
import numpy as np
from MLP import MLP
import sklearn as sklearn
import scipy.stats.mstats as sp
z = 5
activation = np.tanh(np.log(1 + np.exp(z)))
generator = MLP(num_inputs = 436, num_outputs = 436, num_hl=75,  num_hlnodes=50, af = activation) #g
discriminator = MLP(num_inputs = 436 , num_outputs = 1, num_hl=75,  num_hlnodes=50, af = activation) #d




def train(epochs = 200, *args, K = 1, theta):
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
	training_list = []
	for train_set in args:
			training_list.append(train_set)
	m = mean_vectors(*args)
	cov_matrix = np.cov(*args)
	noise_prior = np.random.normal(m, cov_matrix)
	num_training_sets = len(m)
	for i in range(0, len(epochs) - 1):
		momentum_vectors_g = []
		momentum_vectors_d = []
		for j in range(0, K):
			delta_x = []
			J = 0
			for k in range(0, num_training_sets):
				a = training_list[j -1]*generator.weight[j]
				J = J + (np.log(discriminator.predict(training_list)) + (1 - np.log(discriminator.predict((generator.predict(noise_prior))))))
				for l in range(k, 1):
					if k == l:
						for e in range(0, num_training_sets):
							little_delta = little_delta.append(training_list[e])
					else:
						transposed_matrix = (training_list[i] * generator.weight[i -1]).transpose()
						little_delta[i] = little_delta[i + 1]*transposed_matrix*generator.weight[k]
				for f in range(0, len(generator.weight[j])):
					delta_x[k] = (little_delta[k] * training_list[k - 1]).transpose()/discriminator.predict(training_list[i])
		for f in range(0, len(delta)):
			momentum_vectors_g[f] = delta_x[f]
		print(J)
		delta_z = []
		for j in range(0, num_training_sets):
			for k in range(0, len(discriminator.weight)):
				
