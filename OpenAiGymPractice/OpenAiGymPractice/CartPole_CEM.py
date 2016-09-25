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