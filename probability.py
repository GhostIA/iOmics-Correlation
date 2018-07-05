import pandas as pd
import numpy as np
from MLP import MLP
import sklearn as sklearn
import scipy.stats.mstats as sp
z = 5
activation = lambda z: np.tanh(np.log(1 + np.exp(z)))
generator = MLP(num_inputs = 463, num_outputs = 463, num_hl=75,  num_hlnodes=463, af = activation) #g
discriminator = MLP(num_inputs = 463 , num_outputs = 1, num_hl=75,  num_hlnodes=463, af = activation) #d

df = pd.read_excel("C:\\Users\\Groot\\Documents\\cleaned_up_cohort.xlsx")
df1 = df.iloc[:,19:28]#19 to 28
numpy_matrix = df1.as_matrix()



def train(matrix, epochs = 200, K = 1):
	def mean_vectors(matrix):
		training_list = []
		vector_sums = []
		length_of_vectors = matrix.shape[1]
		average_m = [] 
		arr = []
		for i in range(0, length_of_vectors):
			arr = matrix[:,i]
			sum_ = 0
			for j in range(0, len(arr)):
				sum_ = sum_ + arr[j]
			vector_sums.append(sum_)
			
		for i in range(0, len(vector_sums)):
			average = vector_sums[i]/length_of_vectors
			average_m.append(average)
		return average_m

	training_list = matrix
	
	m = mean_vectors(training_list)


	cov_matrix = np.cov(matrix.transpose())
	
	noise_prior = np.random.multivariate_normal(m, cov_matrix)
	num_training_sets = len(m)
	
	print(noise_prior)



	for i in range(0, epochs - 1):
		momentum_vectors_g = []
		momentum_vectors_d = []
		
		for j in range(0, K):
			delta = []
			J = 0
			little_delta = 0
			for k in range(1,matrix.shape[1]):
				a = matrix[:,k - 1]
				z = noise_prior[1]
				W = generator.weights[k]
				x = activation(W * a)
				J = J + (np.log(discriminator.predict(a)) + (1 - np.log(discriminator.predict(generator.predict(z)))))
			for k in range(463, 1, -1):
				for l in range(0, len(noise_prior)): 
					tp = (activation(noise_prior[l] + 0.3))/(2* 0.3)
				if i == l:
					little_delta = generator.weights[l]
				else:
					little_delta = little_delta + tp * (generator.weights[l] * matrix[:,l - 1]).transpose() * generator.weights[l]
			for k in range(len(generator.weights), 0, -1):
				delta_ = (little_delta * matrix[:,l -1])/discriminator.predict(matrix[:,l])
				delta.append(delta_)
			for k in range(0, len(delta)):
				new_momentum = delta[k] * 0.1
				for e in range(0, len(new_momentum)):
					momentum_vectors_d.append(new_momentum[e])

		delta_n = []
		tiny_delta = 0
		for j in range(0, len(noise_prior)):
			for k in range(0, len(discriminator.weights)):
				a = activation((discriminator.weights[k] * noise_prior[j]))
			for k in range(0, len(discriminator.weights)):
				tiny_delta = tiny_delta + tp *(discriminator.weights[k] * a).transpose() * discriminator.weights[k]
				delta_n.append((tiny_delta * a)/(1 - discriminator.predict(generator.predict(a.transpose()))))
			for k in range(0, len(delta_n)):
				new_momentum = delta_n[k] * 0.1
				for e in range(0, len(new_momentum)):
					momentum_vectors_g.append(new_momentum[e])
				for e in range(0,len(generator.weights)):
					generator.weights[e] = generator.weights[e] + 0.1*(delta_n[e]) + 0.1*(momentum_vectors_g[e])
					print(generator.weights[e])
	
train(numpy_matrix)
