import gym
import numpy
import random
import matplotlib.pyplot as plt


print("CartPole_CEM_LinearCombination")
means = [0, 0, 0, 0]
standard_deviations = [100, 100, 100, 100]
weights = []


Episode_Count = 0
env = gym.make("CartPole-v0")
Reward_Array = []
Iteration_Count = 0
for i_episode in range(5000):
	Episode_Count += 1
	observation = env.reset()
	TotalReward = 0
	Weight_Array = [numpy.random.normal(means[0], standard_deviations[0]), 
			numpy.random.normal(means[1], standard_deviations[1]),
			numpy.random.normal(means[2], standard_deviations[2]),
			numpy.random.normal(means[3], standard_deviations[3])]
	for t in range(1000):
		Iteration_Count += 1
		env.render()
		Action =int(numpy.dot(Weight_Array,observation)>0)
		observation, reward, done, info = env.step(Action)
		TotalReward += 1
		
		if done:
			Weight_Array.append(TotalReward)
			Reward_Array.append(TotalReward)
			weights.append(Weight_Array)
			if Episode_Count % 20 == 0:
				Twentieth_Percentile = numpy.percentile(Reward_Array, 80)
				W1 = []
				W2 = []
				W3 = []
				W4 = []
				for W in weights:
					if W[4] >= Twentieth_Percentile:
						W1.append(W[0])
						W2.append(W[1])
						W3.append(W[2])
						W4.append(W[3])
				means[0] = numpy.mean(W1)
				means[1] = numpy.mean(W2)
				means[2] = numpy.mean(W3)
				means[3] = numpy.mean(W4)
				standard_deviations[0] = numpy.std(W1) + numpy.max([5 - (Iteration_Count/10), 0])
				standard_deviations[1] = numpy.std(W2) + numpy.max([5 - (Iteration_Count/10), 0])
				standard_deviations[2] = numpy.std(W3) + numpy.max([5 - (Iteration_Count/10), 0])
				standard_deviations[3] = numpy.std(W4) + numpy.max([5 - (Iteration_Count/10), 0])
				Reward_Array = []
			print("Episode {} finished after {} timesteps".format(Episode_Count, t+1))
			break
		

#import gym
#import numpy
#from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, SoftmaxLayer
#from copy import copy


##create the network and layers
#net = FeedForwardNetwork()
#inp = LinearLayer(4)
#h1 = SigmoidLayer(5)
#h2 = SigmoidLayer(5)
#h3 = SigmoidLayer(5)
#outp = LinearLayer(3)

##add layers to net
#net.addInputModule(inp)
#net.addModule(h1)
#net.addModule(h2)
#net.addModule(h3)
#net.addOutputModule(outp)

##create connections between layers
#in_to_hidden1 = FullConnection(inp, h1)
#hidden1_to_hidden2 = FullConnection(h1, h2)
#hidden2_to_hidden3 = FullConnection(h2, h3)
#hidden3_to_out = FullConnection(h1, outp)

##add connections to net
#net.addConnection(in_to_hidden1)
#net.addConnection(hidden1_to_hidden2)
#net.addConnection(hidden2_to_hidden3)
#net.addConnection(hidden3_to_out)

##used to prepare net for activation
#net.sortModules()

##setup list of means for each weight
#means = numpy.full(len(net.params) - 1, 0)
##setup list of stds for each weight
#standard_deviations = numpy.full(len(net.params) - 1, 100)

#Total_Reward_List = []


#Episode_Count = 0
#env = gym.make("Acrobot-v0")
#Episode_Weights_List = []
#Reward_Array = []
#Iteration_Count = 0
#for i_episode in range(5000):
#	Episode_Count += 1
#	observation = env.reset()
#	TotalReward = 0
#	for i in range(len(net.params) - 1):
#		net.params[i] = numpy.random.normal(means[i], standard_deviations[i] + numpy.max([5 - (Iteration_Count/10), 0]))
		
#	net.sortModules()
#	for t in range(100):
#		Iteration_Count += 1
#		env.render()
		
#		Action = numpy.argmax(net.activate(observation))

#		observation, reward, done, info = env.step(Action)
#		TotalReward += reward
		
#		if done or t > 500:
#			Episode_Weights_List.append([list(net.params), TotalReward])
#			Reward_Array.append(TotalReward)
#			Total_Reward_List.append(TotalReward)
#			if Episode_Count % 20 == 0:
#				Twentieth_Percentile = numpy.percentile(Reward_Array, 80)
#				Elite_Set = [[] for _ in range(len(net.params) - 1)]
#				for W in Episode_Weights_List:
#					if W[1] >= Twentieth_Percentile:
#						for i in range(len(net.params) - 1):
#							Elite_Set[i].append(W[0][i])
#				for i in range(len(net.params) - 1):
#					means[i] = numpy.mean(Elite_Set[i])
#					standard_deviations[i] = numpy.std(Elite_Set[i])
#				Reward_Array = []
#			print("Episode {} finished after {} timesteps. Average Total Reward {}".format(Episode_Count, t+1, numpy.mean(Total_Reward_List)))
#			break
		